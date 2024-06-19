#pragma once
#include <iostream>
#include <cstring>
#include <string>
#include <sstream>
#include <fstream>
#include <windows.h>

#include <cpu.hpp>
#include <kernel.cuh>
#include <filesystem>

#include "Time/Timer.hpp"


// alphabet ranges 
#define ALPHABET_START 'A'
#define ALPHABET_END 'Z'

// computation modes
#define CPU 0
#define GPU 1

using namespace std;

bool readArguments();
bool readInput();