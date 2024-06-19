#include "main.hpp"

string pattern;
string text;
string output_path;
string input_path;

string config_path;
int compute_mode = 0;

int main(int argc, char** argv)
{
	// checking whether any arguments were passed
	if (argc > 1)
	{
		cerr << "Invalid argument: '" << argv[1] << "'." << endl;
		cerr << "Program should be executed without passing any arguments." << endl;
		return 1;
	}

	// when arguments are passed
	if (readArguments() && readInput())
	{
		stringstream transformations;
		int distance = 0;
		if (compute_mode == CPU)
		{
			LevenshteinCPU::init(pattern, text);
			{
				Timer timer;
				distance = LevenshteinCPU::computeLevenshteinDistance();
				cout << "CPU: " << distance << endl;
			}
			transformations = LevenshteinCPU::extractTransformations();
		}
		else
		{
			LevenshteinGPU::init(pattern, text, ALPHABET_START, ALPHABET_END);
			{
				Timer timer;
				distance = LevenshteinGPU::computeLevenshteinDistance();
				cout << "GPU: " << distance << endl;
			}
			transformations = LevenshteinGPU::extractTransformations();
			LevenshteinGPU::end();
		}
		// writing transformations into output file
		ofstream output(output_path);
		if (output.is_open())
		{
			output << distance << endl << transformations.str();
			output.close();
			return 0;
		}
		else
		{
			cout << "Could not open the output file" << endl;
		}
	}
	return 1;
}

// function that checks whether the passed string is the command to exit the program
void shouldExit(string& command)
{
	// exit command equals 'q'
	if (command.size() == 1 && command[0] == 'q')
		exit(0);
}

// function that reads the arguments from the console
bool readArguments()
{
	string buffer;
		
	// reading input path
	while (true)
	{
		cout << "Enter input path: ";
		cin >> buffer;
		shouldExit(buffer);
		if (std::filesystem::is_regular_file(buffer))
		{
			input_path = buffer;
			break;
		}
		cout << "Invalid input path. No such file exists or is in invalid format." << endl;
	}
		
	// reading output path
	while (true)
	{
		cout << "Enter output path: ";
		cin >> buffer;
		shouldExit(buffer);
		ofstream output(buffer);
		if (output.is_open())
		{
			output.close();
			output_path = buffer;
			break;
		}
		cout << "Invalid output path. No such file exists or is in invalid format." << endl;
	}

	// reading computation mode
	while (true)
	{
		cout << "Enter computation mode [g/c]: ";
		cin >> buffer;
		shouldExit(buffer);
		if (buffer.size() == 1)
		{
			switch (buffer[0])
			{
			case 'c':
				compute_mode = CPU;
				break;
			case 'g':
				compute_mode = GPU;
				break;
			default: 
				cout << "Invalid computation mode." << endl;
				continue;
			}			
			break;
		}
		cout << "Invalid computation mode." << endl;
	}
	return true;
}

bool validateInput(string s)
{
	for (int i = 0; i < s.size(); i++)
		if (s[i] < ALPHABET_START || s[i] > ALPHABET_END)
			return false;
	return true;
}

bool readInput()
{
	ifstream input(input_path);
	if (input.is_open())
	{
		input >> pattern;
		input >> text;
		input.close();
		if (!validateInput(pattern) || !validateInput(text))
		{
			cout << "Input is in invalid format. The strings should contain ASCII signs from " 
				<< ALPHABET_START << " to " << ALPHABET_END  << "." << endl;
			return false;
		}
		return true;
	}
	cout << "Could not open the input file" << endl;
	return false;
}
