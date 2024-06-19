#pragma once
#include "cpu.hpp"

vector<vector<int>> distances;
string first_string;
string second_string;
bool empty_string_passed = false;
int index_base = 0;

// function that initializes essential memory
bool LevenshteinCPU::init(string& first, string& second)
{
	first_string = first;
	second_string = second;
	index_base = max(max(first_string.size(), second_string.size()) + 1, NUM_OF_MOVES_TYPES);

	distances = vector<vector<int>>(first.size() + 1);
	for (vector<vector<int>>::iterator it = distances.begin(); it != distances.end(); it++)
	{
		*it = vector<int>(second.size() + 1);
	}
	return true;
}


// function that computes Levenshtein distances array
int LevenshteinCPU::computeLevenshteinDistance()
{
	// filling first row
	for (int j = 0; j <= second_string.size(); j++)
		distances[0][j] = j;

	for (int i = 1; i <= first_string.size(); i++)
	{
		distances[i][0] = i;
		for (int j = 1; j <= second_string.size(); j++)
		{
			if (first_string[i - 1] == second_string[j - 1])
				distances[i][j] = distances[i - 1][j - 1];
			else
				distances[i][j] = std::min(std::min(distances[i - 1][j - 1], distances[i - 1][j]), distances[i][j - 1]) + 1;
		}
	}
	return distances[first_string.size()][second_string.size()];
}

// function that codes transformations that need to be done to 
// transform first string into the second one
string getTransformationCode(int i, int j, int move)
{
	stringstream stream;
	switch (move)
	{
	case MOVE_CHANGE:
		stream << "C:" << i - 1 << ":" << second_string[j - 1];
		break;
	case MOVE_ADD:
		stream << "I:" << i << ":" << second_string[j - 1];
		break;
	case MOVE_DELETE:
		stream << "R:" << i - 1;
		break;
	default:
		CPUError("Invalid move");
	}
	stream << endl;
	return stream.str();
}

// function that extracts transformations from computed D array
stringstream LevenshteinCPU::extractTransformations()
{
	stringstream sstream = stringstream();
	int i = first_string.size();
	int j = second_string.size();
	while (i > -1 && j > -1)
	{
		if (i > 0)
		{
			if (j > 0)
			{
				if (first_string[i - 1] == second_string[j - 1])
				{
					// case [0]
					i--;
					j--;
				}
				else
				{
					if (distances[i - 1][j - 1] <= distances[i - 1][j])
					{
						if (distances[i - 1][j - 1] <= distances[i][j - 1])
						{
							// minimum on diagonal
							sstream << getTransformationCode(i, j, MOVE_CHANGE);
							;							i--;
							j--;
						}
						else
						{
							sstream << getTransformationCode(i, j, MOVE_ADD);
							j--;
						}
					}
					else
					{
						if (distances[i - 1][j] < distances[i][j - 1])
						{
							sstream << getTransformationCode(i, j, MOVE_DELETE);
							i--;
						}
						else
						{
							sstream << getTransformationCode(i, j, MOVE_ADD);
							j--;
						}
					}
				}
			}
			else
			{
				sstream << getTransformationCode(i, j, MOVE_DELETE);
				i--;
			}
		}
		else
		{
			if (j > 0)
			{
				sstream << getTransformationCode(i, j, MOVE_ADD);
			}
			j--;
		}
	}
	return sstream;
}


