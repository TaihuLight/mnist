#include "lab_parser.hpp"

#include <algorithm>
#include <cassert>

using std::begin;
using std::copy;
using std::end;
using std::min;
using std::runtime_error;

using arma::Col;
using arma::join_cols;

namespace mnist {

LabelParser::LabelParser(const char *filename)
  : MnistParser<Col<uint8_t>>{filename}
{
  // Read label file header into buffer
  if (!file_.read(reinterpret_cast<char *>(buffer_), kHeaderSizeLabelFile)) {
    throw runtime_error{"Could not read label file header."};
  }

  // Validate magic number
  if (ReadBigEndianInt32(buffer_) != kMagicNumberLabelFile) {
    throw runtime_error{"Unexpected magic number for label file."};
  }

  // Read number of labels
  num_items_ = ReadBigEndianInt32(buffer_ + 4);
}

Col<uint8_t> LabelParser::Parse()
{
  if (IsDone()) {
    throw runtime_error{"No more labels available."};
  }
  Col<uint8_t> col(num_items_);
  auto it = begin(col);
  do {
    const int cnt = min((int)kBufferSize, num_items_);
    if (!file_.read(reinterpret_cast<char *>(buffer_), cnt)) {
      throw runtime_error{"Could not read label."};
    }
    num_items_ -= cnt;
    copy(buffer_, buffer_ + cnt, it);
    it += cnt;
  } while (num_items_ != 0);
  assert(it == end(col));
  return col;
}

} // namespace mnist
