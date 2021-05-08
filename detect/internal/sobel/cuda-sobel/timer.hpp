#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <iostream>

class timer {
    using clock      = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<clock>;
    using us         = std::chrono::microseconds;  // 10^(-6) of a second

   public:
    timer() { start_time_point = clock::now(); }

    ~timer() {
        finish_time_point = clock::now();

        auto start_time  = std::chrono::time_point_cast<us>( start_time_point).time_since_epoch().count();
        auto finish_time = std::chrono::time_point_cast<us>(finish_time_point).time_since_epoch().count();

        auto duration = finish_time - start_time;
        double millisecond = duration * 0.001;

        std::cout << "Time spent: " << duration << "us (" << millisecond << "ms)" << std::endl;
    }

   private:
    time_point start_time_point;
    time_point finish_time_point;
};

#endif // TIMER_HPP
