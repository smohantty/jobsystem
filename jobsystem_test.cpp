#include <jobsystem.hpp>
#include <iostream>

#include <atomic>

using namespace jobsystem;

struct User {
    std::atomic_int calls = {0};
    void func(JobSystem&, JobSystem::Job*) {
        calls++;
    };
};

int main(int argc, char **argv) {
    if(argc != 1) {
        std::cout << argv[0] << " takes no arguments.\n";
        return 1;
    }
    User user;
    jobsystem::JobSystem js;
    //js.adopt();

    JobSystem::Job* root = js.createJob<User, &User::func>(nullptr, &user);
    for (int i=0 ; i<256 ; i++) {
        JobSystem::Job* job = js.createJob<User, &User::func>(root, &user);
        js.run(job, JobSystem::DONT_SIGNAL);
    }
    js.runAndWait(root);
    js.emancipate();
    std::cout<<"The value is :"<<user.calls;

    return 0;
}
