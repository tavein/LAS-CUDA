/*
 * DebugOutput.cpp
 *
 *  Created on: Feb 7, 2015
 *      Author: versus
 */

#include <chrono>
#include <string>

#include "DebugOutput.h"

void DebugOutput::log(const std::string& message)
{
    auto t =
            std::chrono::duration_cast < std::chrono::milliseconds
                    > (std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    std::cout << (t / (1000 * 60)) % 60 << ":" << (t / 1000) % 60 << "."
            << t % 1000 << " - " << message << std::endl;
}
