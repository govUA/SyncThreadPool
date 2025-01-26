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

    std::atomic<size_t> totalWaitingTime{0};
    std::atomic<size_t> totalTasksExecuted{0};
    std::atomic<size_t> totalTaskQueueLength{0};
    std::atomic<size_t> queueSamples{0};

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
        condition.wait(lock, [this] { return taskQueue.empty(); });
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
        taskQueue.emplace([this, task]() {
            auto start = std::chrono::steady_clock::now();
            (*task)();
            auto end = std::chrono::steady_clock::now();
            totalTasksExecuted++;
            totalWaitingTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        });
    }
    condition.notify_one();
    return {taskId, std::move(result)};
}

void ThreadPool::workerThread() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            totalTaskQueueLength += taskQueue.size();
            queueSamples++;

            condition.wait(lock, [this] { return stopFlag || (!pauseFlag && !taskQueue.empty()); });
            if (stopFlag && taskQueue.empty()) return;
            if (pauseFlag) continue;

            task = std::move(taskQueue.front());
            taskQueue.pop();

            if (taskQueue.empty()) {
                condition.notify_all();
            }
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

    auto startTime = std::chrono::steady_clock::now();
    for (int i = 1; i <= numTasks; ++i) {
        auto [id, future] = pool.addTask(simulateTask, i);
        taskIds.push_back(id);
        results.push_back(std::move(future));
    }

    std::cout << "Pausing the thread pool for 5 seconds...\n";
    pool.pause();
    std::this_thread::sleep_for(std::chrono::seconds(5));

    std::cout << "Resuming the thread pool.\n";
    pool.resume();

    for (auto &result: results) {
        result.get();
    }

    auto endTime = std::chrono::steady_clock::now();
    pool.waitAndStop();

    auto totalTime = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    size_t avgQueueLength = pool.queueSamples ? pool.totalTaskQueueLength / pool.queueSamples : 0;
    size_t avgTaskExecutionTime = pool.totalTasksExecuted ? pool.totalWaitingTime / pool.totalTasksExecuted : 0;

    std::cout << "Total time: " << totalTime << " seconds\n";
    std::cout << "Average queue length: " << avgQueueLength << "\n";
    std::cout << "Average task execution time: " << avgTaskExecutionTime << " milliseconds\n";
    std::cout << "Total tasks executed: " << pool.totalTasksExecuted << "\n";
    return 0;
}
