#pragma once
#if defined _WIN32 || defined __CYGWIN__
  #ifdef BUILDING_JOBSYSTEM
    #define JOBSYSTEM_PUBLIC __declspec(dllexport)
  #else
    #define JOBSYSTEM_PUBLIC __declspec(dllimport)
  #endif
#else
  #ifdef BUILDING_JOBSYSTEM
      #define JOBSYSTEM_PUBLIC __attribute__ ((visibility ("default")))
  #else
      #define JOBSYSTEM_PUBLIC
  #endif
#endif

#include <functional>
#include <atomic>
#include <memory>

namespace jobsystem
{

class JOBSYSTEM_PUBLIC JobSystem
{
  static constexpr size_t CACHELINE_SIZE = 64;

public:
    class Job;
    struct JobSystemImpl;

    using JobFunc = void(*)(void*, JobSystem&, Job*);

    class alignas(CACHELINE_SIZE) Job
    {
    public:
        Job() noexcept = default;
        Job(const Job&) = delete;
        Job(Job&&) = delete;

    private:
        friend class JobSystem;
        friend struct JobSystemImpl;

        // Size is chosen so that we can store at least std::function<>
        // the alignas() qualifier ensures we're multiple of a cache-line.
        static constexpr size_t JOB_STORAGE_SIZE_BYTES =
                sizeof(std::function<void()>) > 48 ? sizeof(std::function<void()>) : 48;
        static constexpr size_t JOB_STORAGE_SIZE_WORDS =
                (JOB_STORAGE_SIZE_BYTES + sizeof(void*) - 1) / sizeof(void*);

        // keep it first, so it's correctly aligned with all architectures
        // this is were we store the job's data, typically a std::function<>
                                                                // v7 | v8
        void* mStorage[JOB_STORAGE_SIZE_WORDS];                  // 48 | 48
        JobFunc mFunction;                                       //  4 |  8
        uint16_t mParent;                                        //  2 |  2
        std::atomic<uint16_t> mRunningJobCount = { 1 };          //  2 |  2
        mutable std::atomic<uint16_t> mRefCount = { 1 };         //  2 |  2
                                                                //  6 |  2 (padding)
                                                                // 64 | 64
    };


  explicit JobSystem(size_t threadCount = 0, size_t adoptableThreadsCount = 1) noexcept;
  ~JobSystem();


    // Make the current thread part of the thread pool.
    void adopt();

    // Remove this adopted thread from the parent. This is intended to be used for
    // shutting down a JobSystem. In particular, this doesn't allow the parent to
    // adopt more thread.
    void emancipate();


    // If a parent is not specified when creating a job, that job will automatically take the
    // root job as a parent.
    // The root job is reset when waited on.
    Job* setRootJob(Job* job) noexcept;


  Job* create(Job* parent, JobFunc func) noexcept;


    // NOTE: All methods below must be called from the same thread and that thread must be
    // owned by JobSystem's thread pool.

    /*
     * Job creation examples:
     * ----------------------
     *
     *  struct Functor {
     *   uintptr_t storage[6];
     *   void operator()(JobSystem&, Jobsystem::Job*);
     *  } functor;
     *
     *  struct Foo {
     *   uintptr_t storage[6];
     *   void method(JobSystem&, Jobsystem::Job*);
     *  } foo;
     *
     *  Functor and Foo size muse be <= uintptr_t[6]
     *
     *   createJob()
     *   createJob(parent)
     *   createJob<Foo, &Foo::method>(parent, &foo)
     *   createJob<Foo, &Foo::method>(parent, foo)
     *   createJob<Foo, &Foo::method>(parent, std::ref(foo))
     *   createJob(parent, functor)
     *   createJob(parent, std::ref(functor))
     *   createJob(parent, [ up-to 6 uintptr_t ](JobSystem*, Jobsystem::Job*){ })
     *
     *  Utility functions:
     *  ------------------
     *    These are less efficient, but handle any size objects using the heap if needed.
     *    (internally uses std::function<>), and don't require the callee to take
     *    a (JobSystem&, Jobsystem::Job*) as parameter.
     *
     *  struct BigFoo {
     *   uintptr_t large[16];
     *   void operator()();
     *   void method(int answerToEverything);
     *   static void exec(BigFoo&) { }
     *  } bigFoo;
     *
     *   jobs::createJob(js, parent, [ any-capture ](int answerToEverything){}, 42);
     *   jobs::createJob(js, parent, &BigFoo::method, &bigFoo, 42);
     *   jobs::createJob(js, parent, &BigFoo::exec, std::ref(bigFoo));
     *   jobs::createJob(js, parent, bigFoo);
     *   jobs::createJob(js, parent, std::ref(bigFoo));
     *   etc...
     *
     *  struct SmallFunctor {
     *   uintptr_t storage[3];
     *   void operator()(T* data, size_t count);
     *  } smallFunctor;
     *
     *   jobs::parallel_for(js, data, count, [ up-to 3 uintptr_t ](T* data, size_t count) { });
     *   jobs::parallel_for(js, data, count, smallFunctor);
     *   jobs::parallel_for(js, data, count, std::ref(smallFunctor));
     *
     */

    // creates an empty (no-op) job with an optional parent
    Job* createJob(Job* parent = nullptr) noexcept {
        return create(parent, nullptr);
    }

    // creates a job from a KNOWN method pointer w/ object passed by pointer
    // the caller must ensure the object will outlive the Job
    template<typename T, void(T::*method)(JobSystem&, Job*)>
    Job* createJob(Job* parent, T* data) noexcept {
        struct stub {
            static void call(void* user, JobSystem& js, Job* job) noexcept {
                (*static_cast<T**>(user)->*method)(js, job);
            }
        };
        Job* job = create(parent, &stub::call);
        if (job) {
            job->mStorage[0] = data;
        }
        return job;
    }

    // creates a job from a KNOWN method pointer w/ object passed by value
    template<typename T, void(T::*method)(JobSystem&, Job*)>
    Job* createJob(Job* parent, T data) noexcept {
        static_assert(sizeof(data) <= sizeof(Job::mStorage), "user data too large");
        struct stub {
            static void call(void* user, JobSystem& js, Job* job) noexcept {
                T* that = static_cast<T*>(user);
                (that->*method)(js, job);
                that->~T();
            }
        };
        Job* job = create(parent, &stub::call);
        if (job) {
            new(job->mStorage) T(std::move(data));
        }
        return job;
    }

    // creates a job from a functor passed by value
    template<typename T>
    Job* createJob(Job* parent, T functor) noexcept {
        static_assert(sizeof(functor) <= sizeof(Job::mStorage), "functor too large");
        struct stub {
            static void call(void* user, JobSystem& js, Job* job) noexcept {
                T& that = *static_cast<T*>(user);
                that(js, job);
                that.~T();
            }
        };
        Job* job = create(parent, &stub::call);
        if (job) {
            new(job->mStorage) T(std::move(functor));
        }
        return job;
    }

    /*
     * Jobs are normally finished automatically, this can be used to cancel a job before it is run.
     *
     * Never use this once a flavor of run() has been called.
     */
    void cancel(Job*& job) noexcept;

    /*
     * Adds a reference to a Job.
     *
     * This allows the caller to waitAndRelease() on this job from multiple threads.
     * Use runAndWait() if waiting from multiple threads is not needed.
     *
     * This job MUST BE waited on with waitAndRelease(), or released with release().
     */
    Job* retain(Job* job) noexcept;

    /*
     * Releases a reference from a Job obtained with runAndRetain() or a call to retain().
     *
     * The job can't be used after this call.
     */
    void release(Job*& job) noexcept;
    void release(Job*&& job) noexcept {
        Job* p = job;
        release(p);
    }

    /*
     * Add job to this thread's execution queue. It's reference will drop automatically.
     * Current thread must be owned by JobSystem's thread pool. See adopt().
     *
     * The job can't be used after this call.
     */
    enum runFlags { DONT_SIGNAL = 0x1 };
    void run(Job*& job, uint32_t flags = 0) noexcept;
    void run(Job*&& job, uint32_t flags = 0) noexcept { // allows run(createJob(...));
        Job* p = job;
        run(p);
    }

    void signal() noexcept;

    /*
     * Add job to this thread's execution queue and and keep a reference to it.
     * Current thread must be owned by JobSystem's thread pool. See adopt().
     *
     * This job MUST BE waited on with wait(), or released with release().
     */
    Job* runAndRetain(Job* job, uint32_t flags = 0) noexcept;

    /*
     * Wait on a job and destroys it.
     * Current thread must be owned by JobSystem's thread pool. See adopt().
     *
     * The job must first be obtained from runAndRetain() or retain().
     * The job can't be used after this call.
     */
    void waitAndRelease(Job*& job) noexcept;

    /*
     * Runs and wait for a job. This is equivalent to calling
     *  runAndRetain(job);
     *  wait(job);
     *
     * The job can't be used after this call.
     */
    void runAndWait(Job*& job) noexcept;
    void runAndWait(Job*&& job) noexcept { // allows runAndWait(createJob(...));
        Job* p = job;
        runAndWait(p);
    }

public:
  int get_number() const;
private:
  std::unique_ptr<JobSystemImpl> mImpl;
};

}

