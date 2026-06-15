
#include<iostream>
#include<vector>

/*
    Pseudo random number using the logistic random chaotic system

    
*/

//the recursive relation is given by 
// x[n+1] = r *  x[n] * (x[n] + 1)
struct initialKey{
    double r;
};

std::vector<double> pseudoRandomStream(const initialKey& key , int randomNumbersRequried){
    std::vector<double> randomStream(randomNumbersRequried);

    constexpr int numberOfIterationsRequriedForStability{0};

    double random{1};
    for(size_t i = 1; i <= numberOfIterationsRequriedForStability; ++i){
        random = key.r * random * (1 - random);
    }
    return randomStream;
}

void printRandoms(){
    //print the random numbers generated using for all the values of r in the range 
    //of 2 to 10 and then the iteration count form 100 to 500
}

int main(){
    printRandoms();
    return 0;
}