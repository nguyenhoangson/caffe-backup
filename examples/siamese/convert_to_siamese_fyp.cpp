//
// This script converts MNIST-like dataset to the leveldb format used
// by caffe to train siamese network.
// Usage:
//    convert_to_siamese_fyp iconic_image_file iconic_label_file insitu_image_file insitu_label_file output_db_file

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_LEVELDB
#include "leveldb/db.h"

/* UTILITY  */
uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

/* Read index_th image data  */
void read_image(std::ifstream* image_file, 
		std::ifstream* label_file,
		uint32_t index, uint32_t rows, 
		uint32_t cols, const uint32_t num_channels,
		char* pixels, char* label) {
  
  /* seekg(): sets the position of the next character to be extracted 
     from the input stream. E.g: 
   */
  image_file->seekg(index * num_channels * rows * cols + 16);
  image_file->read(pixels, num_channels * rows * cols);
  label_file->seekg(index + 8);
  label_file->read(label, 1);
}

/* CONVERT IMAGE TO LEVELDB FILES  */
// void convert_dataset(const char* image_filename, 
// 		     const char* label_filename,
// 		     const char* db_filename) {
  
//   // Open files
//   std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
//   std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
//   CHECK(image_file) << "Unable to open file " << image_filename;
//   CHECK(label_file) << "Unable to open file " << label_filename;
  
//   // Read the magic and the meta data
//   uint32_t magic;
//   uint32_t num_items;
//   uint32_t num_labels;
//   uint32_t rows;
//   uint32_t cols;
  
//   // validate data
//   image_file.read(reinterpret_cast<char*>(&magic), 4);
//   magic = swap_endian(magic);

//   // TODO: delete this
//   std::cout << "Magic: " << magic << std::endl; 

//   CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
//   label_file.read(reinterpret_cast<char*>(&magic), 4);
//   magic = swap_endian(magic);
//   CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
//   image_file.read(reinterpret_cast<char*>(&num_items), 4);
//   num_items = swap_endian(num_items);
//   label_file.read(reinterpret_cast<char*>(&num_labels), 4);
//   num_labels = swap_endian(num_labels);
//   CHECK_EQ(num_items, num_labels);
//   image_file.read(reinterpret_cast<char*>(&rows), 4);
//   rows = swap_endian(rows);
//   image_file.read(reinterpret_cast<char*>(&cols), 4);
//   cols = swap_endian(cols);

//   // Open leveldb
//   leveldb::DB* db;
//   leveldb::Options options;
//   options.create_if_missing = true;
//   options.error_if_exists = true;
//   leveldb::Status status = leveldb::DB::Open(
//       options, db_filename, &db);
//   CHECK(status.ok()) << "Failed to open leveldb " << db_filename
//       << ". Is it already existing?";

//   char label_i;
//   char label_j;
//   char* pixels = new char[2 * rows * cols];
//   std::string value;
  
//   /* set up caffe datum: a google protobuf messsage  
//      used to store data and optionally a label 
//      datum = 3D matrix of (width, height, channel, label(optional))
//    */
//   caffe::Datum datum;
//   datum.set_channels(2);  // one channel for each image in the pair
//   datum.set_height(rows); // height of image
//   datum.set_width(cols);  // width of image
//   LOG(INFO) << "A total of " << num_items << " items.";
//   LOG(INFO) << "Rows: " << rows << " Cols: " << cols;

//   // Write to leveldb file
//   for (int itemid = 0; itemid < num_items; ++itemid) {
//     int i = caffe::caffe_rng_rand() % num_items;  // pick a random  pair
//     int j = caffe::caffe_rng_rand() % num_items;
    
//     // add image data to datum 
//     read_image(&image_file, &label_file, i, rows, cols,
//         pixels, &label_i);
//     read_image(&image_file, &label_file, j, rows, cols,
//         pixels + (rows * cols), &label_j);
//     datum.set_data(pixels, 2*rows*cols);
    
//     // if two images file have the same label => y = 0
//     if (label_i  == label_j) {
//       datum.set_label(1);
//     } else {
//       datum.set_label(0);
//     }
    
//     // Write to level db file 
//     datum.SerializeToString(&value);
//     std::string key_str = caffe::format_int(itemid, 8);
//     db->Put(leveldb::WriteOptions(), key_str, value);
//   }

//   delete db;
//   delete [] pixels;
// }


void convert_to_siamese_data(const char* iconic_image_filename, 
			     const char* iconic_label_filename,
			     const char* insitu_image_filename, 
			     const char* insitu_label_filename,
			     const char* db_output_filename) {
  
  // Open files
  std::ifstream iconic_image_file(iconic_image_filename, std::ios::in | std::ios::binary);
  std::ifstream iconic_label_file(iconic_label_filename, std::ios::in | std::ios::binary);
  std::ifstream insitu_image_file(insitu_image_filename, std::ios::in | std::ios::binary);
  std::ifstream insitu_label_file(insitu_label_filename, std::ios::in | std::ios::binary);
  CHECK(iconic_image_filename) << "Unable to open file " << iconic_image_filename;
  CHECK(iconic_label_filename) << "Unable to open file " << iconic_label_filename;
  CHECK(insitu_image_filename) << "Unable to open file " << insitu_image_filename;
  CHECK(insitu_label_filename) << "Unable to open file " << insitu_label_filename;

  // Read the magic and the meta data
  uint32_t iconic_magic;
  uint32_t insitu_magic;
  uint32_t num_iconic_items;
  uint32_t num_iconic_labels;
  uint32_t num_insitu_items;
  uint32_t num_insitu_labels;
  const uint32_t num_channels = 3;  

  // we make assumption that iconic and insitu images are in the same shape
  uint32_t rows;
  uint32_t cols;
  
  // validate data in iconic files
  iconic_image_file.read(reinterpret_cast<char*>(&iconic_magic), 4);
  iconic_magic = swap_endian(iconic_magic);
  CHECK_EQ(iconic_magic, 2051) << "Incorrect image file magic.";
  iconic_label_file.read(reinterpret_cast<char*>(&iconic_magic), 4);
  iconic_magic = swap_endian(iconic_magic);
  CHECK_EQ(iconic_magic, 2049) << "Incorrect label file magic.";
  iconic_image_file.read(reinterpret_cast<char*>(&num_iconic_items), 4);
  num_iconic_items = swap_endian(num_iconic_items);
  iconic_label_file.read(reinterpret_cast<char*>(&num_iconic_labels), 4);
  num_iconic_labels = swap_endian(num_iconic_labels);
  CHECK_EQ(num_iconic_items, num_iconic_labels);
  iconic_image_file.read(reinterpret_cast<char*>(&rows), 4);
  rows = swap_endian(rows);
  iconic_image_file.read(reinterpret_cast<char*>(&cols), 4);
  cols = swap_endian(cols);
  
  // validate data in insitu files
  insitu_image_file.read(reinterpret_cast<char*>(&insitu_magic), 4);
  insitu_magic = swap_endian(insitu_magic);
  CHECK_EQ(insitu_magic, 2051) << "Incorrect image file magic.";
  insitu_label_file.read(reinterpret_cast<char*>(&insitu_magic), 4);
  insitu_magic = swap_endian(insitu_magic);
  CHECK_EQ(insitu_magic, 2049) << "Incorrect label file magic.";
  insitu_image_file.read(reinterpret_cast<char*>(&num_insitu_items), 4);
  num_insitu_items = swap_endian(num_insitu_items);
  insitu_label_file.read(reinterpret_cast<char*>(&num_insitu_labels), 4);
  num_insitu_labels = swap_endian(num_insitu_labels);
  CHECK_EQ(num_insitu_items, num_insitu_labels);
  
  // TODO: Delete this 
  std::cout << "Row: " << rows << std::endl;
  std::cout << "Col: " << cols << std::endl;
  
  // TODO: delete this
  std::cout << "Magic: " << insitu_magic << std::endl; 

  // open leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, db_output_filename, &db);
  CHECK(status.ok()) << "Failed to open train leveldb " << db_output_filename
      << ". Is it already existing?";
  
  char label_iconic;
  char label_insitu;
  char* pixels = new char[6 * rows * cols];
  std::string value;
  
  /* set up caffe datum: a google protobuf messsage  
     used to store data and optionally a label 
     datum = 3D matrix of (width, height, channel, label(optional))
   */
  caffe::Datum datum;
  datum.set_channels(6);  // 3 channels for each RGB image in the pair
  datum.set_height(rows); // height of image
  datum.set_width(cols);  // width of image
  LOG(INFO) << "A total of " << num_iconic_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  

  // Write to train_leveldb file
  for (int insitu_id = 0; insitu_id < num_insitu_items; ++insitu_id) {
    
    // read image from insitu 
    read_image(&insitu_image_file, &insitu_label_file, 
	       insitu_id, rows, cols, num_channels, 
	       pixels, &label_insitu);
    
    for(int iconic_id = 0; iconic_id < num_iconic_items; ++iconic_id){

        read_image(&iconic_image_file, &iconic_label_file, 
		   iconic_id, rows, cols, num_channels,
		   pixels + (num_channels * rows * cols), &label_iconic);
    
        datum.set_data(pixels, 6*rows*cols);
    
        // if two images file have the same label => y = 0
        if (label_insitu  == label_iconic) {
            datum.set_label(1);
        } else {
            datum.set_label(0);
        }
    }
    
    // Write to level db file 
    datum.SerializeToString(&value);
    std::string key_str = caffe::format_int(insitu_id, 8);
    db->Put(leveldb::WriteOptions(), key_str, value);
  }
  
  delete db;
  delete [] pixels;
}


int main(int argc, char** argv) {
  if (argc != 6) {
    printf("This script converts the MNIST-like dataset to the leveldb format used\n"
           "by caffe to train a siamese network.\n"
           "Usage:\n"
           "    convert_to_siamese_fyp iconic_image_file iconic_label_file insitu_image_file insitu_label_file output_db_file "
           "\n"
           );
  } else {
    google::InitGoogleLogging(argv[0]);
    std::cout << "Testing" << std::endl;
    convert_to_siamese_data(argv[1], argv[2], argv[3], argv[4], argv[5]);
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires LevelDB; compile with USE_LEVELDB.";
}
#endif  // USE_LEVELDB
