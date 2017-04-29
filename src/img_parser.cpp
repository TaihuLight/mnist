#include "img_parser.hpp"

#include <algorithm>

using std::min;
using std::runtime_error;

using arma::Mat;
using arma::Row;

namespace mnist {


ImageParser::ImageParser(const char * filename)
  : MnistParser<Mat<uint8_t>>{filename}
{
  // Read image file header into buffer
  if (!file_.read(reinterpret_cast<char *>(buffer_), kHeaderSizeImageFile)) {
    throw runtime_error{"Could not read image file header."};
  }

  // Validate magic number
  if (ReadBigEndianInt32(buffer_) != kMagicNumberImageFile) {
    throw runtime_error{"Unexpected magic number for image file."};
  }

  // Read numbers of images and of rows and columns per image
  num_items_ = ReadBigEndianInt32(buffer_ + 4);
  num_row_   = ReadBigEndianInt32(buffer_ + 8);
  num_col_   = ReadBigEndianInt32(buffer_ + 12);
}

Mat<uint8_t> ImageParser::Parse()
{
  if (IsDone()) {
    throw runtime_error{"File already parsed."};
  }
  Mat<uint8_t> mat(num_items_, kImageSize);
  for (int i = 0; --num_items_ != 0; ++i) {
    if (!file_.read(reinterpret_cast<char *>(buffer_), kImageSize)) {
      throw runtime_error{"Could not read image."};
    }
    Row<uint8_t> row(buffer_, kImageSize);
    mat.row(i) = row;
  }
  return mat;
}

} // namespace mnist
