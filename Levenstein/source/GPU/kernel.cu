#include "kernel.cuh"

#define BLOCK_SIZE 32
#define WARP_SIZE min(BLOCK_SIZE, warpSize)
#define WARPS_PER_BLOCK (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE 

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

bool strings_are_swapped = false;

// device arrays
int* dev_distances_glob = nullptr;
int* dev_accessory_glob = nullptr;

// device strings
char* dev_pattern_glob = nullptr;
char* dev_text_glob = nullptr;

// alphabet variables
char alphabet_first_symbol_glob;
char alphabet_size_glob;

string pattern_glob;
string text_glob;

// error checking
void checkCUDAError(const char* msg, int line = -1) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// function which was only used to debug the code
// it copies a device 2D array to host and print it in specified mode
// which describes whether the array is stored in the column or the row order
void printPartOfDevArray(void* dev_arr, int byte_size, int type, int mode = ROW_MODE, int row_size = -1)
{
	if (!dev_arr || byte_size < 0 || row_size < -1) return;

	void* host_void_array = nullptr;
	int word_length = 1;
	switch (type)
	{
	case INT_ARR:
		word_length = sizeof(int);
		break;
	case CHAR_ARR:
		word_length = sizeof(char);
		break;
	default:
		return;
	}

	host_void_array = malloc(byte_size);
	if (!host_void_array)
		return;

	cudaMemcpy(host_void_array, dev_arr, byte_size, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy host_void_array");

	int words_count = byte_size / word_length;
	int rows = row_size == -1 ? words_count : words_count / row_size;
	int columns = row_size == -1 ? words_count : row_size;
	int k = 0;
	for (int i = 0; i < rows; i++)
	{
		if (mode == COL_MODE)
			k = i;
		for (int j = 0; j < columns && k < words_count; j++)
		{
			switch (type)
			{
			case INT_ARR:
				printf("%d ", ((int*)host_void_array)[k]);
				break;
			case CHAR_ARR:
				printf("%c ", ((char*)host_void_array)[k]);
				break;
			default:
				return;
			}
			if (mode == COL_MODE)
			{
				k += rows;
			}
			else
			{
				k++;
			}
		}
		printf("\n");
	}
	fflush(stdout);
	free(host_void_array);
}

// function that initializes all necessary memory on the gpu
bool LevenshteinGPU::init(string& first, string& second, char alphabet_first_symbol, char alphabet_last_symbol)
{
	alphabet_first_symbol_glob = alphabet_first_symbol;
	alphabet_size_glob = alphabet_last_symbol - alphabet_first_symbol + 1;

	// swapping strings to increase number of working threads
	strings_are_swapped = first.size() > second.size();
	if (first.size() <= second.size())
	{
		pattern_glob = string(first);
		text_glob = string(second);
	} 
	else
	{
		pattern_glob = string(second);
		text_glob = string(first);
	}

	int pattern_size = pattern_glob.size();
	int text_size = text_glob.size();
	int array_size = (pattern_size + 1) * (text_size + 1);
	

	// allocating buffers
	cudaMalloc((void**)&dev_distances_glob, array_size * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc error");

	cudaMalloc((void**)&dev_pattern_glob, pattern_size * sizeof(char));
	checkCUDAErrorWithLine("cudaMalloc error");

	cudaMalloc((void**)&dev_text_glob, text_size * sizeof(char));
	checkCUDAErrorWithLine("cudaMalloc error");

	cudaMalloc((void**)&dev_accessory_glob, alphabet_size_glob * (text_size + 1) * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc error");

	// coping strings into GPU memory
	cudaMemcpy((void*)dev_pattern_glob, (void*)pattern_glob.c_str(), pattern_size * sizeof(char), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcpy error");

	cudaMemcpy((void*)dev_text_glob, (void*)text_glob.c_str(), text_size * sizeof(char), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcpy error");

	return true;
}

// kernel which computes X array
__global__ void kernFillAccessoryArray(int alphabet_size, int text_size, char* text, int* accessory_array, char alphabet_first_symbol)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < alphabet_size)
	{
		int i = index * (text_size + 1);
		int j = 0;
		char symbol = alphabet_first_symbol + index;

		accessory_array[i] = 0;
		i++;

		while (j < text_size)
		{
			accessory_array[i] = text[j] == symbol ? j + 1 : accessory_array[i - 1];
			i++;
			j++;
		}		
	}
}

// part of kernel which implements determining the value of currently computed array element
// based on precalculated Avar, Bvar, etc.
// it is basically extracted code from kernLevenshteinSingleRow
__device__ void computeLevenshteinCell(int text_size, char* pattern, char* text,
	int* distances, int* accessory_array, int iteration, int index, char alphabet_first_symbol,
	int i, int Avar, int Bvar)
{
	// computing values of l and X[l,j]
	int l = pattern[iteration - 1] - alphabet_first_symbol;
	int X = accessory_array[l * (text_size + 1) + index];

	// case of first column 
	if (index == 0)
	{
		distances[i] = iteration;
	}
	// the rest of array
	else
	{
		// case when currently processed symbols of pattern and text are equal
		if (pattern[iteration - 1] == text[index - 1])
		{
			distances[i] = Avar;
		}
		else
		{
			// case when X[l,j] == 0
			if (X == 0)
			{
				distances[i] = min(min(Avar, Bvar), iteration + index - 1) + 1;
			}
			else
			{
				int Cvar = distances[(iteration - 1) * (text_size + 1) + X - 1];
				distances[i] = min(min(Avar, Bvar), Cvar + index - 1 - X) + 1;
			}
		}
	}
}

// function that releases whole allocated memory
void LevenshteinGPU::end()
{
	cudaFree(dev_distances_glob);
	cudaFree(dev_accessory_glob);
	cudaFree(dev_pattern_glob);
	cudaFree(dev_text_glob);
}

// kernel that performs computation of single row in D array
__global__ void kernLevenshteinSingleRow(int pattern_size, int text_size,
	char* pattern, char* text,
	int* distances, int* accessory_array,
	char alphabet_first_symbol, int iteration)
{
	// thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index <= text_size && iteration <= pattern_size) // iteration represents i and index represents j in the article
	{
		int i = iteration * (text_size + 1) + index;

		if (iteration == 0)
		{
			distances[i] = index;
			return;
		}

		int Avar = 0;
		if(index > 0)
			Avar = distances[i - (text_size + 1) - 1];
		int Bvar = distances[i - (text_size + 1)];

		computeLevenshteinCell(text_size, pattern, text, distances, accessory_array,
			iteration, index, alphabet_first_symbol, i, Avar, Bvar);
	}
}

// function that computes X array described in the article
void fillAccessoryArray()
{
	int accessory_number_of_blocks = (alphabet_size_glob + BLOCK_SIZE) / BLOCK_SIZE;
	kernFillAccessoryArray << < accessory_number_of_blocks, BLOCK_SIZE >> > (alphabet_size_glob, text_glob.size(), dev_text_glob, dev_accessory_glob, alphabet_first_symbol_glob);
	checkCUDAErrorWithLine("kernComputeLevenshteinArray");

	cudaDeviceSynchronize();
	checkCUDAErrorWithLine("synchronization");
}

// function that copies the last element in computed distance array form GPU
// and returns it 
int getResult()
{
	int result = -1;
	cudaMemcpy(&result, &(dev_distances_glob[(pattern_glob.size() + 1) * (text_glob.size() + 1) - 1]), sizeof(int), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("copying result shared and shuffle");
	return result;
}

// function that computes Levenshtein distance with shared memory
int singleRow()
{
	int number_of_blocks = (text_glob.size() + BLOCK_SIZE) / BLOCK_SIZE;
	for (int i = 0; i <= pattern_glob.size(); i++)
	{
		kernLevenshteinSingleRow << <number_of_blocks, BLOCK_SIZE >> > (pattern_glob.size(), text_glob.size(),
			dev_pattern_glob, dev_text_glob, dev_distances_glob, dev_accessory_glob, alphabet_first_symbol_glob, i);
		checkCUDAErrorWithLine("shared memory kernel");

		cudaDeviceSynchronize();
		checkCUDAErrorWithLine("synchronization");
	}

	return getResult();
}


// computing Levenshtein distance by the use of sepecified algorithm
int LevenshteinGPU::computeLevenshteinDistance()
{
	fillAccessoryArray();

	return singleRow();
}

// mapping 2D indices to 1D index
int indicesToIndex(int i, int j, int row_size)
{
	return i * row_size + j;
}

// generating code for transformation
string getTransformationCodeGPU(int i, int j, int move)
{
	stringstream stream;
	
	switch (move)
	{
	case GPU_MOVE_CHANGE:
		stream << "C:" << i - 1 << ":" << j - 1;
		break;
	case GPU_MOVE_ADD:
		stream << "I:" << i << ":" << j - 1;
		break;
	case GPU_MOVE_DELETE:
		stream << "R:" << i - 1;
		break;
	default:
		break;
	}
	stream << endl;
	return stream.str();
}

// function that extracts transformations from D array
stringstream LevenshteinGPU::extractTransformations()
{
	int* distances = new int[(pattern_glob.size() + 1) * (text_glob.size() + 1)];
	stringstream sstream = stringstream();
	if (!distances)
		return sstream;
	int i = pattern_glob.size();
	int j = text_glob.size();
	while (i > -1 && j > -1)
	{
		if (i > 0)
		{
			// case when change operation is allowed
			if (j > 0)
			{
				// case when the symbols of both strings are the same
				if (pattern_glob[i - 1] == text_glob[j - 1])
				{
					// nothing to transform
					i--;
					j--;
				}
				else
				{
					// indices are mapped as follows:
					// i-1,j-1 -> ind11
					// i-1,j -> ind10
					// i,j-1 -> ind01
					int ind11 = indicesToIndex(i - 1, j - 1, text_glob.size() + 1);
					int ind10 = indicesToIndex(i - 1, j, text_glob.size() + 1);
					int ind01 = indicesToIndex(i, j - 1, text_glob.size() + 1);
					if (distances[ind11] <= distances[ind10])
					{
						// case when D[i-1,j-1] is the minimal value among the considered ones
						if (distances[ind11] <= distances[ind01])
						{
							// depending on swapping the change transformation is performed from pattern or from text
							if (!strings_are_swapped)
							{

								sstream << getTransformationCodeGPU(i, j, GPU_MOVE_CHANGE);
							}
							else
							{
								sstream << getTransformationCodeGPU(j, i, GPU_MOVE_CHANGE);
							}
							i--;
							j--;
						}
						// case when D[i,j-1] is the minimal value among the considered ones
						else
						{
							// depending on swapping the change transformation is performed from pattern or from text
							if (!strings_are_swapped)
							{
								sstream << getTransformationCodeGPU(i, j, GPU_MOVE_ADD);
							}
							else
							{
								sstream << getTransformationCodeGPU(j, i, GPU_MOVE_DELETE);
							}
							j--;
						}
					}
					else
					{
						// case when D[i-1,j] is the minimal value among the considered ones
						if (distances[ind10] < distances[ind01])
						{
							// depending on swapping the change transformation is performed from pattern or from text
							if (!strings_are_swapped)
							{
								sstream << getTransformationCodeGPU(i, j, GPU_MOVE_DELETE);
							}
							else
							{
								sstream << getTransformationCodeGPU(j, i, GPU_MOVE_ADD);
							}
							i--;
						}
						// case when D[i,j-1] is the minimal value among the considered ones
						else
						{
							// depending on swapping the change transformation is performed from pattern or from text
							if (!strings_are_swapped)
							{
								sstream << getTransformationCodeGPU(i, j, GPU_MOVE_ADD);
							}
							else
							{
								sstream << getTransformationCodeGPU(j, i, GPU_MOVE_DELETE);
							}
							j--;
						}
					}
				}
			}
			// case when change operation is not allowed
			else
			{
				// depending on swapping the change transformation is performed from pattern or from text
				if (!strings_are_swapped)
				{
					sstream << getTransformationCodeGPU(i, j, GPU_MOVE_DELETE);
				}
				else
				{
					sstream << getTransformationCodeGPU(j, i, GPU_MOVE_ADD);
				}
				i--;
			}
		}
		// case when change operation is not allowed
		else
		{
			// depending on swapping the change transformation is performed from pattern or from text
			if (j > 0)
			{
				if (!strings_are_swapped)
				{
					sstream << getTransformationCodeGPU(i, j, GPU_MOVE_ADD);
				}
				else
				{
					sstream << getTransformationCodeGPU(j, i, GPU_MOVE_DELETE);
				}
			}
			j--;
		}
	}
	delete distances;
	return sstream;

}