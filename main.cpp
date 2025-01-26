#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <unordered_map>
#include <chrono>
#include <random>

class ThreadPool {
public:
    ThreadPool(size_t numThreads);

    ~ThreadPool();

    template<typename Func, typename... Args>
    auto addTask(Func &&func,
                 Args &&... args) -> std::pair<int, std::future<typename std::invoke_result<Func, Args...>::type>>;

    void stop();
    void pause();
    void resume();

    void waitAndStop();

private:
    void workerThread();

    using Task = std::function<void()>;

    std::vector<std::thread> workers;
    std::queue<Task> taskQueue;
    std::mutex queueMutex;
    std::condition_variable condition;
    std::atomic<bool> stopFlag = false;
    std::atomic<bool> pauseFlag = false;

    std::atomic<int> taskIdCounter = 0;
    std::unordered_map<int, std::future<void>> taskResults;
};

ThreadPool::ThreadPool(size_t numThreads) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back(&ThreadPool::workerThread, this);
    }
}

ThreadPool::~ThreadPool() {
    waitAndStop();
}

void ThreadPool::stop() {
    stopFlag = true;
    condition.notify_all();
    for (std::thread &worker: workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::pause() {
    pauseFlag = true;
}

void ThreadPool::resume() {
    pauseFlag = false;
    condition.notify_all();
}

void ThreadPool::waitAndStop() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        condition.wait(lock, [this] { return stopFlag || (!pauseFlag && !taskQueue.empty()); });
    }
    stop();
}

template<typename Func, typename... Args>
auto ThreadPool::addTask(Func &&func,
                         Args &&... args) -> std::pair<int, std::future<typename std::invoke_result<Func, Args...>::type>> {
    using ReturnType = typename std::invoke_result<Func, Args...>::type;

    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            [func = std::forward<Func>(func), ...args = std::forward<Args>(args)]() mutable {
                return func(std::forward<Args>(args)...);
            });

    std::future<ReturnType> result = task->get_future();
    int taskId = ++taskIdCounter;

    {
        std::unique_lock<std::mutex> lock(queueMutex);
        if (stopFlag) {
            throw std::runtime_error("ThreadPool is stopping, cannot add new tasks.");
        }
        taskQueue.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return {taskId, std::move(result)};
}

void ThreadPool::workerThread() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            condition.wait(lock, [this] { return stopFlag || !taskQueue.empty(); });
            if (stopFlag && taskQueue.empty()) return;

            task = std::move(taskQueue.front());
            taskQueue.pop();
        }
        task();
    }
}

void simulateTask(int id) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(5, 10);
    int sleepTime = dist(gen);

    std::cout << "Task " << id << " started, taking " << sleepTime << " seconds.\n";
    std::this_thread::sleep_for(std::chrono::seconds(sleepTime));
    std::cout << "Task " << id << " completed.\n";
}

int main() {
    ThreadPool pool(4);

    const int numTasks = 10;
    std::vector<std::future<void>> results;
    std::vector<int> taskIds;

    for (int i = 1; i <= numTasks; ++i) {
        auto [id, future] = pool.addTask(simulateTask, i);
        taskIds.push_back(id);
        results.push_back(std::move(future));
    }

    for (auto &result: results) {
        result.get();
    }

    pool.waitAndStop();

    std::cout << "All tasks completed. Thread pool stopped.\n";
    return 0;
}
