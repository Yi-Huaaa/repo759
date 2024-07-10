#include <iostream>

int main(int argc, char* argv[]){
	int N = atoi(argv[1]);
	for (int i = 0; i < (N+1); i++)
		printf("%d ", i);
	printf("\n");

	for (int i = N; i > -1; i--)
		std::cout << i << " ";
	std::cout << std::endl;
	
	return 0;
}