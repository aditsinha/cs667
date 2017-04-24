
#include <iostream>
#include <fstream>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Must provide features, labels, and number of examples\n";
    exit(1);
  }

  int num_examples = std::atoi(argv[3]);
  
  std::ifstream feature_file(argv[1]);
  std::ifstream label_file(argv[2]);

  // read over the preamble of the feature file
  for (int i = 0; i < 16; i++) {
    feature_file.get();
  }

  // read over the preamble of the label file
  for (int i = 0; i < 8; i++) {
    label_file.get();
  }

  for (int i = 0; i < num_examples; i++) {
    // first output the label.  Divide into < 5 and >= 5.
    int label = label_file.get();
    std::cout << (int)(label < 5) << ",";
    
    // each image is 28*28
    for (int j = 0; j < 28*28; j++) {
      int pval = feature_file.get();
      std::cout << pval << ",";
    }
    std::cout << std::endl;
  }

  return 0;
}
