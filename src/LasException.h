#pragma once

#include <exception>
#include <string>

class LasException : public std::exception
{

    const std::string message;

public:
    LasException(const std::string& message);
    virtual ~LasException();

    virtual const char* what() const noexcept;
};

