#ifndef MNIST_PARSER_H_
#define MNIST_PARSER_H_

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace mnist {

template <typename T>
class MnistParser {
public:
                    MnistParser(const MnistParser&) = delete;
                    MnistParser(MnistParser&&) = delete;
  virtual           ~MnistParser() = default;
  MnistParser&      operator=(const MnistParser&) = delete;
  bool              IsDone() const;
  virtual T         Parse() = 0;
protected:
  enum {
    kMagicNumberLabelFile = 0x801,
    kMagicNumberImageFile = 0x803,
    kHeaderSizeLabelFile = 8,
    kHeaderSizeImageFile = 16,
    kImageSize = 784 // 24 x 24 matrix
  };
  explicit          MnistParser(const char *);
  static int        ReadBigEndianInt32(const uint8_t *);
  std::ifstream     file_;
  int               num_items_;
};

template <typename T>
MnistParser<T>::MnistParser(const char * filename)
  : file_{filename, std::ios::in | std::ios::binary}, num_items_(0)
{
  if (!file_) {
    throw std::runtime_error{std::string("File not found: ") + filename};
  }
}

template <typename T>
inline bool MnistParser<T>::IsDone() const
{
  return num_items_ == 0;
}

template <typename T>
inline int MnistParser<T>::ReadBigEndianInt32(const uint8_t * bytes)
{
  return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

} // namespace mnist

#endif // MNIST_PARSER_H_
