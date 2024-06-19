#pragma once
#include <iostream>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>
#include <stack>

// Exiting with error
#define CPUError(msg) (fprintf(stderr,"Error occurred [%d]: %s\n",__LINE__,msg), exit(EXIT_FAILURE))

#define NUM_OF_MOVES_TYPES (size_t)3
#define MOVE_CHANGE 0
#define MOVE_DELETE 1
#define MOVE_ADD 2

using namespace std;

namespace LevenshteinCPU
{
	bool init(string& first, string& second);
	int computeLevenshteinDistance();
	stringstream extractTransformations();
	
}