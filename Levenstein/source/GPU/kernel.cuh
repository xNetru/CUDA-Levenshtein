#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
//#include "thrust/sort.h"
//#include "thrust/device_ptr.h"
//#include "thrust/iterator/zip_iterator.h"
//#include "thrust/scan.h"

#include <iostream>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>
#include <stack>
#include <algorithm>
#include <stdlib.h>

#define INT_ARR 0
#define CHAR_ARR 1

#define COL_MODE 0
#define ROW_MODE 1

#define GPU_MOVE_CHANGE 0
#define GPU_MOVE_DELETE 1
#define GPU_MOVE_ADD 2


using namespace std;

namespace LevenshteinGPU 
{
	bool init(string& first, string& second, char alphabet_first_symbol, char alphabet_last_symbol);
	int computeLevenshteinDistance();
	stringstream extractTransformations();
	void end();
}