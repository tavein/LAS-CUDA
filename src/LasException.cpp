/*
 * LasException.cpp
 *
 *  Created on: Apr 5, 2015
 *      Author: versus
 */

#include "LasException.h"

LasException::LasException(const std::string& message)
        : message(message)
{

}

LasException::~LasException()
{
}

const char* LasException::what() const noexcept
{
    return message.c_str();
}
