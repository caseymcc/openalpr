cmake_minimum_required (VERSION 3.2)

include("cmake/HunterGate.cmake")

HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.103.tar.gz"
    SHA1 "0b8c31bfdf0fd2fa87b4f743aabcd73904afa256"
)

project(OpenAlpr)

add_subdirectory(src)