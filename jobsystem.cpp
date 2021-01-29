#include <jobsystem.hpp>

#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <random>


#include <iostream>

#include "Allocator.h"
#include "WorkStealingDequeue.h"
#include "robin_map.h"

namespace jobsystem {

static constexpr size_t CACHELINE_SIZE = 64;
static constexpr size_t MAX_JOB_COUNT = 4096;
static_assert(MAX_JOB_COUNT <= 0x7FFE, "MAX_JOB_COUNT must be <= 0x7FFE");

using WorkQueue = utils::WorkStealingDequeue<uint16_t, MAX_JOB_COUNT>;

struct alignas(CACHELINE_SIZE) ThreadState
{    // this causes 40-bytes padding
    // make sure storage is cache-line aligned
    WorkQueue workQueue;

    // these are not accessed by the worker threads
    alignas(CACHELINE_SIZE)     // this causes 56-bytes padding
    JobSystem*  js;
    std::thread thread;
    std::default_random_engine rndGen;
    uint32_t id;
};

struct JobSystem::JobSystemImpl
{
    JobSystemImpl(JobSystem* system, size_t userThreadCount, size_t adoptableThreadsCount)
    : mJobPool("JobSystem Job pool", MAX_JOB_COUNT * sizeof(Job)),
      mJobStorageBase(static_cast<Job *>(mJobPool.getAllocator().getCurrent())),
      mSystem(system)
    {
        int threadPoolCount = userThreadCount;
        if (threadPoolCount == 0) {
            // default value, system dependant
            int hwThreads = std::thread::hardware_concurrency();
            // if (UTILS_HAS_HYPER_THREADING) {
            //     // For now we avoid using HT, this simplifies profiling.
            //     // TODO: figure-out what to do with Hyper-threading
            //     // since we assumed HT, always round-up to an even number of cores (to play it safe)
            //     hwThreads = (hwThreads + 1) / 2;
            // }
            // make sure we have at least one thread in the thread pool
            hwThreads = std::max(2, hwThreads);
            // one of the thread will be the user thread
            threadPoolCount = hwThreads - 1;
        }
        threadPoolCount = std::min(UTILS_HAS_THREADING ? 32 : 0, threadPoolCount);

        mThreadStates = aligned_vector<ThreadState>(threadPoolCount + adoptableThreadsCount);
        mThreadCount = uint16_t(threadPoolCount);
        mParallelSplitCount = (uint8_t)std::ceil((std::log2f(threadPoolCount + adoptableThreadsCount)));

        // this is a pity these are not compile-time checks (C++17 supports it apparently)
        assert(mExitRequested.is_lock_free());
        assert(Job().mRunningJobCount.is_lock_free());

        std::random_device rd;
        const size_t hardwareThreadCount = mThreadCount;
        auto& states = mThreadStates;

        #pragma nounroll
        for (size_t i = 0, n = states.size(); i < n; i++) {
            auto& state = states[i];
            state.rndGen = std::default_random_engine(rd());
            state.id = (uint32_t)i;
            state.js = nullptr;
            if (i < hardwareThreadCount) {
                // don't start a thread of adoptable thread slots
                state.thread = std::thread(&JobSystemImpl::loop, this, &state);
            }
        }
    }

    ~JobSystemImpl() {
        requestExit();

        #pragma nounroll
        for (auto &state : mThreadStates) {
            // adopted threads are not joinable
            if (state.thread.joinable()) {
                state.thread.join();
            }
        }
    }

    void release(JobSystem::Job*& job) noexcept {
        decRef(job);
        job = nullptr;
    }

    JobSystem::Job* retain(JobSystem::Job* job) noexcept
    {
        JobSystem::Job* retained = job;
        incRef(retained);
        return retained;
    }

    void loop(ThreadState* state) noexcept {
        //setThreadName("JobSystem::loop");
        //setThreadPriority(Priority::DISPLAY);

        // set a CPU affinity on each of our JobSystem thread to prevent them from jumping from core
        // to core. On Android, it looks like the affinity needs to be reset from time to time.
        //setThreadAffinityById(state->id);

        // record our work queue
        mThreadMapLock.lock();
        bool inserted = mThreadMap.emplace(std::this_thread::get_id(), state).second;
        mThreadMapLock.unlock();
        //ASSERT_PRECONDITION(inserted, "This thread is already in a loop.");
        log(state->id, "thread started");
        // run our main loop...
        do {
            if (!execute(*state)) {
                std::unique_lock<Mutex> lock(mWaiterLock);
                while (!exitRequested() && !hasActiveJobs()) {
                    wait(lock);
                    //setThreadAffinityById(state->id);
                }
            }
        } while (!exitRequested());
        log(state->id, "thread ended");
    }

    inline bool exitRequested() const noexcept {
        // memory_order_relaxed is safe because the only action taken is to exit the thread
        return mExitRequested.load(std::memory_order_relaxed);
    }

    inline bool hasActiveJobs() const noexcept {
        return mActiveJobs.load(std::memory_order_relaxed) > 0;
    }

    inline bool hasJobCompleted(JobSystem::Job const* job) noexcept {
        return job->mRunningJobCount.load(std::memory_order_relaxed) <= 0;
    }

    bool execute(ThreadState& state) noexcept {
        ////HEAVY_SYSTRACE_CALL();

        Job* job = pop(state.workQueue);
        if (UTILS_UNLIKELY(job == nullptr)) {
            // our queue is empty, try to steal a job
            log(state.id, "steal");
            job = steal(state);
        } else {
            log(state.id, "execute");
        }

        if (job) {
            UTILS_UNUSED_IN_RELEASE
            uint32_t activeJobs = mActiveJobs.fetch_sub(1, std::memory_order_relaxed);
            assert(activeJobs); // whoops, we were already at 0
            //HEAVY_SYSTRACE_VALUE32("JobSystem::activeJobs", activeJobs - 1);

            if (UTILS_LIKELY(job->mFunction)) {
                //HEAVY_SYSTRACE_NAME("job->mFunction");
                job->mFunction(job->mStorage, *mSystem, job);
            }
            finish(job);
        }
        return job != nullptr;
    }


    inline void incRef(Job const* job) noexcept {
        // no action is taken when incrementing the reference counter, therefore we can safely use
        // memory_order_relaxed.
        job->mRefCount.fetch_add(1, std::memory_order_relaxed);
    }

    UTILS_NOINLINE
    void decRef(Job const* job) noexcept {

        // We must ensure that accesses from other threads happen before deleting the Job.
        // To accomplish this, we need to guarantee that no read/writes are reordered after the
        // dec-ref, because ANOTHER thread could hold the last reference (after us) and that thread
        // needs to see all accesses completed before it deletes the object. This is done
        // with memory_order_release.
        // Similarly, we need to guarantee that no read/write are reordered before the last decref,
        // or some other thread could see a destroyed object before the ref-count is 0. This is done
        // with memory_order_acquire.
        auto c = job->mRefCount.fetch_sub(1, std::memory_order_acq_rel);
        assert(c > 0);
        if (c == 1) {
            // This was the last reference, it's safe to destroy the job.
            mJobPool.destroy(job);
        }
    }

    void requestExit() noexcept {
        mExitRequested.store(true);
        { std::lock_guard<Mutex> lock(mWaiterLock); }
        mWaiterCondition.notify_all();

    }

    void wait(std::unique_lock<Mutex>& lock) noexcept {
        ++mWaiterCount;
        mWaiterCondition.wait(lock);
        --mWaiterCount;
    }

    void wake() noexcept {
        Mutex& lock = mWaiterLock;
        lock.lock();
        const uint32_t waiterCount = mWaiterCount;
        lock.unlock();
        //mWaiterCondition.notify_n(waiterCount);
        if (waiterCount == 1) {
            mWaiterCondition.notify_one();
        } else if (waiterCount > 1) {
            mWaiterCondition.notify_all();
        }
    }

    inline ThreadState& getState() noexcept {
        std::lock_guard<utils::SpinLock> lock(mThreadMapLock);
        auto iter = mThreadMap.find(std::this_thread::get_id());
        //ASSERT_PRECONDITION(iter != mThreadMap.end(), "This thread has not been adopted.");
        return *iter->second;
    }

    JobSystem::Job* allocateJob() noexcept {
        return mJobPool.make<Job>();
    }

    inline ThreadState* getStateToStealFrom(ThreadState& state) noexcept {
        auto& threadStates = mThreadStates;
        // memory_order_relaxed is okay because we don't take any action that has data dependency
        // on this value (in particular mThreadStates, is always initialized properly).
        uint16_t adopted = mAdoptedThreads.load(std::memory_order_relaxed);
        uint16_t const threadCount = mThreadCount + adopted;

        ThreadState* stateToStealFrom = nullptr;

        // don't try to steal from someone else if we're the only thread (infinite loop)
        if (threadCount >= 2) {
            do {
                // this is biased, but frankly, we don't care. it's fast.
                uint16_t index = uint16_t(state.rndGen() % threadCount);
                assert(index < threadStates.size());
                stateToStealFrom = &threadStates[index];
                // don't steal from our own queue
            } while (stateToStealFrom == &state);
        }
        return stateToStealFrom;
    }

    JobSystem::Job* steal(ThreadState& state) noexcept {
        Job* job = nullptr;
        do {
            ThreadState* const stateToStealFrom = getStateToStealFrom(state);
            if (UTILS_LIKELY(stateToStealFrom)) {
                job = steal(stateToStealFrom->workQueue);
            }
            // nullptr -> nothing to steal in that queue either, if there are active jobs,
            // continue to try stealing one.
        } while (!job && hasActiveJobs());
        return job;
    }



    UTILS_NOINLINE
    void finish(JobSystem::Job* job) noexcept {
        //HEAVY_SYSTRACE_CALL();

        bool notify = false;

        // terminate this job and notify its parent
        auto* storage = mJobStorageBase;
        do {
            // std::memory_order_release here is needed to synchronize with JobSystem::wait()
            // which needs to "see" all changes that happened before the job terminated.
            auto runningJobCount = job->mRunningJobCount.fetch_sub(1, std::memory_order_acq_rel);
            assert(runningJobCount > 0);
            if (runningJobCount == 1) {
                // no more work, destroy this job and notify its parent
                notify = true;
                auto* parent = job->mParent == 0x7FFF ? nullptr : &storage[job->mParent];
                decRef(job);
                job = const_cast<JobSystem::Job*>(parent);
            } else {
                // there is still work (e.g.: children), we're done.
                break;
            }
        } while (job);

        // wake-up all threads that could potentially be waiting on this job finishing
        if (notify) {
            wake();
        }
    }


    void run(JobSystem::Job*& job, uint32_t flags) noexcept {
        ////HEAVY_SYSTRACE_CALL();

        ThreadState& state(getState());

        // increase the active job count before we add the job to the queue, because otherwise
        // the job could run and finish before the counter is incremented, which would trigger
        // an assert() in execute(). Either way, it's not "wrong", but the assert() is useful.
        uint32_t activeJobs = mActiveJobs.fetch_add(1, std::memory_order_relaxed);

        put(state.workQueue, job);

        //HEAVY_SYSTRACE_VALUE32("JobSystem::activeJobs", activeJobs + 1);

        // wake-up a thread if needed...
        if (!(flags & DONT_SIGNAL)) {
            // wake-up multiple queues because there could be multiple jobs queued
            // especially if DONT_SIGNAL was used
            wake();
        }

        // after run() returns, the job is virtually invalid (it'll die on its own)
        job = nullptr;
    }

    void runAndWait(JobSystem::Job*& job) noexcept
    {
        runAndRetain(job);
        waitAndRelease(job);
    }

    JobSystem::Job* runAndRetain(JobSystem::Job* job, uint32_t flags = 0) noexcept
    {
        JobSystem::Job* retained = retain(job);
        run(job, flags);
        return retained;
    }

    void waitAndRelease(Job*& job) noexcept {
        //SYSTRACE_CALL();

        assert(job);
        assert(job->mRefCount.load(std::memory_order_relaxed) >= 1);

        ThreadState& state(getState());
        do {
            if (!execute(state)) {
                // test if job has completed first, to possibly avoid taking the lock
                if (hasJobCompleted(job)) {
                    break;
                }

                // the only way we can be here is if the job we're waiting on it being handled
                // by another thread:
                //    - we returned from execute() which means all queues are empty
                //    - yet our job hasn't completed yet
                //    ergo, it's being run in another thread
                //
                // this could take time however, so we will wait with a condition, and
                // continue to handle more jobs, as they get added.

                std::unique_lock<Mutex> lock(mWaiterLock);
                if (!hasJobCompleted(job) && !hasActiveJobs() && !exitRequested()) {
                    wait(lock);
                }
            }
        } while (!hasJobCompleted(job) && !exitRequested());

        if (job == mRootJob) {
            mRootJob = nullptr;
        }

        release(job);
    }

    void adopt() {
        const auto tid = std::this_thread::get_id();

        std::unique_lock<utils::SpinLock> lock(mThreadMapLock);
        auto iter = mThreadMap.find(tid);
        ThreadState* const state = iter ==  mThreadMap.end() ? nullptr : iter->second;
        lock.unlock();

        if (state) {
            // we're already part of a JobSystem, do nothing.
            // ASSERT_PRECONDITION(this == state->js,
            //         "Called adopt on a thread owned by another JobSystem (%p), this=%p!",
            //         state->js, this);
            return;
        }

        // memory_order_relaxed is safe because we don't take action on this value.
        uint16_t adopted = mAdoptedThreads.fetch_add(1, std::memory_order_relaxed);
        size_t index = mThreadCount + adopted;

        // ASSERT_POSTCONDITION(index < mThreadStates.size(),
        //         "Too many calls to adopt(). No more adoptable threads!");

        // all threads adopted by the JobSystem need to run at the same priority
        //JobSystem::setThreadPriority(JobSystem::Priority::DISPLAY);

        // This thread's queue will be selectable immediately (i.e.: before we set its TLS)
        // however, it's not a problem since mThreadState is pre-initialized and valid
        // (e.g.: the queue is empty).

        lock.lock();
        mThreadMap[tid] = &mThreadStates[index];
    }

    void put(WorkQueue& workQueue, Job* job) noexcept {
        size_t index = job - mJobStorageBase;
        assert(index >= 0 && index < MAX_JOB_COUNT);
        workQueue.push(uint16_t(index + 1));
    }

    Job* pop(WorkQueue& workQueue) noexcept {
        size_t index = workQueue.pop();
        assert(index <= MAX_JOB_COUNT);
        return !index ? nullptr : &mJobStorageBase[index - 1];
    }

    Job* steal(WorkQueue& workQueue) noexcept {
        size_t index = workQueue.steal();
        assert(index <= MAX_JOB_COUNT);
        return !index ? nullptr : &mJobStorageBase[index - 1];
    }

    void emancipate() {
        const auto tid = std::this_thread::get_id();
        std::lock_guard<utils::SpinLock> lock(mThreadMapLock);
        auto iter = mThreadMap.find(tid);
        //ThreadState* const state = iter ==  mThreadMap.end() ? nullptr : iter->second;
        // ASSERT_PRECONDITION(state, "this thread is not an adopted thread");
        // ASSERT_PRECONDITION(state->js == this, "this thread is not adopted by us");
        mThreadMap.erase(iter);
    }

    JobSystem::Job* create(JobSystem::Job* parent, JobFunc func) noexcept
    {
        parent = (parent == nullptr) ? mRootJob : parent;
        Job* const job = allocateJob();
        if (UTILS_LIKELY(job)) {
            size_t index = 0x7FFF;
            if (parent) {
                // add a reference to the parent to make sure it can't be terminated.
                // memory_order_relaxed is safe because no action is taken at this point
                // (the job is not started yet).
                auto parentJobCount = parent->mRunningJobCount.fetch_add(1, std::memory_order_relaxed);

                // can't create a child job of a terminated parent
                assert(parentJobCount > 0);

                index = parent - mJobStorageBase;
                assert(index < MAX_JOB_COUNT);
            }
            job->mFunction = func;
            job->mParent = uint16_t(index);
        }
        return job;
    }

    void log(uint32_t threadId, const char* str)
    {
        std::lock_guard<std::mutex> lock(mLoggingLock);
        std::cout<<"Thread: "<<threadId<<",\tMsg: "<<str<<"\n";
    }

    std::mutex mLoggingLock;

    // these have thread contention, keep them together
    std::mutex mWaiterLock;
    std::condition_variable mWaiterCondition;
    uint32_t mWaiterCount = 0;

    std::atomic<uint32_t> mActiveJobs = { 0 };
    utils::Arena<utils::ThreadSafeObjectPoolAllocator<JobSystem::Job>, utils::LockingPolicy::NoLock> mJobPool;

    template <typename T>
    using aligned_vector = std::vector<T, utils::STLAlignedAllocator<T>>;

    // these are essentially const, make sure they're on a different cache-lines than the
    // read-write atomics.
    // We can't use "alignas(CACHELINE_SIZE)" because the standard allocator can't make this
    // guarantee.
    char padding[CACHELINE_SIZE];

    alignas(16) // at least we align to half (or quarter) cache-line
    aligned_vector<ThreadState> mThreadStates;          // actual data is stored offline
    std::atomic<bool> mExitRequested = { false };       // this one is almost never written
    std::atomic<uint16_t> mAdoptedThreads = { 0 };      // this one is almost never written
    JobSystem::Job* mJobStorageBase;                         // Base for conversion to indices
    uint16_t mThreadCount = 0;                          // total # of threads in the pool
    uint8_t mParallelSplitCount = 0;                    // # of split allowable in parallel_for
    Job* mRootJob = nullptr;
    JobSystem* mSystem = nullptr;

    utils::SpinLock mThreadMapLock; // this should have very little contention
    tsl::robin_map<std::thread::id, ThreadState *> mThreadMap;
};

JobSystem::JobSystem(size_t threadCount, size_t adoptableThreadsCount) noexcept
:mImpl(std::make_unique<JobSystemImpl>(this, threadCount, adoptableThreadsCount))
{

}

JobSystem::~JobSystem(){}

// -----------------------------------------------------------------------------------------------
// public API...


JobSystem::Job* JobSystem::create(JobSystem::Job* parent, JobFunc func) noexcept
{
    return mImpl->create(parent, func);
}

void JobSystem::cancel(Job*& job) noexcept
{
    mImpl->finish(job);
    job = nullptr;
}

JobSystem::Job* JobSystem::retain(JobSystem::Job* job) noexcept
{
    JobSystem::Job* retained = job;
    mImpl->incRef(retained);
    return retained;
}

void JobSystem::release(JobSystem::Job*& job) noexcept {
    mImpl->decRef(job);
    job = nullptr;
}

void JobSystem::signal() noexcept
{
    mImpl->wake();
}

void JobSystem::run(JobSystem::Job*& job, uint32_t flags) noexcept
{
    mImpl->run(job, flags);
}

JobSystem::Job* JobSystem::runAndRetain(JobSystem::Job* job, uint32_t flags) noexcept
{
    return mImpl->runAndRetain(job, flags);
}

void JobSystem::waitAndRelease(Job*& job) noexcept
{
    mImpl->waitAndRelease(job);
}

void JobSystem::runAndWait(JobSystem::Job*& job) noexcept
{
    mImpl->runAndWait(job);
}

void JobSystem::adopt()
{
    mImpl->adopt();
}

void JobSystem::emancipate()
{
    mImpl->emancipate();
}



int JobSystem::get_number() const
{
  return 0;
}

}
