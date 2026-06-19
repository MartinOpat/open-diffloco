#pragma once
/// @file npz_reader.hpp
/// Minimal .npz / .npy reader supporting numpy format v1.0 AND v2.0.
/// Replaces cnpy which only handles v1.0 and crashes on v2.0 headers.
///
/// Only supports: reading, little-endian float32/float64, C-order arrays.
/// That's all we need for policy deployment.

#include <Eigen/Core>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// minizip / zlib for .npz (which is just a .zip)
#include <zlib.h>

namespace jave {
namespace npz {

// Single array loaded from .npy

struct NpyArray {
  std::vector<char> data;
  std::vector<size_t> shape;
  size_t word_size = 0; // 4 = float32, 8 = float64
  bool is_float = true;

  size_t num_elements() const {
    size_t n = 1;
    for (auto s : shape)
      n *= s;
    return n;
  }

  /// Get element as double (handles f4/f8 transparently).
  double as_double(size_t i) const {
    if (word_size == 8)
      return reinterpret_cast<const double *>(data.data())[i];
    else
      return static_cast<double>(
          reinterpret_cast<const float *>(data.data())[i]);
  }

  /// Load into Eigen VectorXd.
  Eigen::VectorXd to_vector() const {
    const size_t n = num_elements();
    Eigen::VectorXd v(n);
    for (size_t i = 0; i < n; ++i)
      v(static_cast<int>(i)) = as_double(i);
    return v;
  }

  /// Load into Eigen MatrixXd (row-major numpy --> col-major Eigen).
  Eigen::MatrixXd to_matrix() const {
    if (shape.size() != 2)
      throw std::runtime_error("to_matrix: array is not 2-D");
    const int rows = static_cast<int>(shape[0]);
    const int cols = static_cast<int>(shape[1]);
    Eigen::MatrixXd m(rows, cols);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        m(r, c) = as_double(r * cols + c);
    return m;
  }

  /// Load scalar (0-D or single-element array).
  double to_scalar() const { return as_double(0); }
};

// Parse a .npy blob (v1.0 or v2.0)

inline NpyArray parse_npy(const char *buf, size_t len) {
  // Magic: \x93NUMPY
  if (len < 10 || buf[0] != '\x93' || std::memcmp(buf + 1, "NUMPY", 5) != 0)
    throw std::runtime_error("parse_npy: not a valid .npy buffer");

  uint8_t major = static_cast<uint8_t>(buf[6]);
  // uint8_t minor = static_cast<uint8_t>(buf[7]);

  uint32_t header_len = 0;
  size_t header_offset = 0;

  if (major == 1) {
    // v1.0: 2-byte little-endian header length at offset 8
    header_len = static_cast<uint16_t>(static_cast<uint8_t>(buf[8]) |
                                       (static_cast<uint8_t>(buf[9]) << 8));
    header_offset = 10;
  } else if (major >= 2) {
    // v2.0+: 4-byte little-endian header length at offset 8
    header_len = static_cast<uint32_t>(static_cast<uint8_t>(buf[8]) |
                                       (static_cast<uint8_t>(buf[9]) << 8) |
                                       (static_cast<uint8_t>(buf[10]) << 16) |
                                       (static_cast<uint8_t>(buf[11]) << 24));
    header_offset = 12;
  } else {
    throw std::runtime_error("parse_npy: unsupported npy major version");
  }

  if (header_offset + header_len > len)
    throw std::runtime_error("parse_npy: header extends past buffer");

  std::string header(buf + header_offset, header_len);
  const char *data_start = buf + header_offset + header_len;
  size_t data_len = len - header_offset - header_len;

  // Parse header dict:  {'descr': '<f8', 'fortran_order': False, 'shape':
  // (49,), }
  NpyArray arr;

  // dtype
  std::regex descr_re("'descr'\\s*:\\s*'([^']*)'");
  std::smatch m;
  if (!std::regex_search(header, m, descr_re))
    throw std::runtime_error("parse_npy: no descr in header");
  std::string descr = m[1].str();

  if (descr == "<f8" || descr == "=f8" || descr == "f8")
    arr.word_size = 8;
  else if (descr == "<f4" || descr == "=f4" || descr == "f4")
    arr.word_size = 4;
  else if (descr == "<i8" || descr == "<i4" || descr == "<u8" || descr == "<u4")
    // integer scalars (n_hidden, etc.)
    arr.word_size = (descr.back() == '8') ? 8 : 4;
  else
    throw std::runtime_error("parse_npy: unsupported dtype: " + descr);

  arr.is_float = (descr.find('f') != std::string::npos);

  // shape
  std::regex shape_re("'shape'\\s*:\\s*\\(([^)]*)\\)");
  if (!std::regex_search(header, m, shape_re))
    throw std::runtime_error("parse_npy: no shape in header");
  std::string shape_str = m[1].str();
  // Parse comma-separated ints (may be empty for 0-d, or "49," for 1-d)
  {
    std::istringstream ss(shape_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      // Trim whitespace
      tok.erase(0, tok.find_first_not_of(" \t\n"));
      tok.erase(tok.find_last_not_of(" \t\n") + 1);
      if (!tok.empty())
        arr.shape.push_back(std::stoull(tok));
    }
  }

  // Copy data
  size_t expected = arr.num_elements() * arr.word_size;
  if (data_len < expected)
    throw std::runtime_error("parse_npy: data truncated (expected " +
                             std::to_string(expected) + " bytes, got " +
                             std::to_string(data_len) + ")");

  // For integer types stored as scalars, convert to double in a float64 buffer
  if (!arr.is_float) {
    arr.data.resize(arr.num_elements() * 8);
    for (size_t i = 0; i < arr.num_elements(); ++i) {
      double val = 0;
      if (arr.word_size == 8) {
        int64_t iv;
        std::memcpy(&iv, data_start + i * 8, 8);
        val = static_cast<double>(iv);
      } else {
        int32_t iv;
        std::memcpy(&iv, data_start + i * 4, 4);
        val = static_cast<double>(iv);
      }
      std::memcpy(arr.data.data() + i * 8, &val, 8);
    }
    arr.word_size = 8;
    arr.is_float = true;
  } else {
    arr.data.assign(data_start, data_start + expected);
  }

  return arr;
}

// Load .npz (zip of .npy files)

/// Reads .npz via the ZIP central directory (handles data descriptors
/// from np.savez_compressed where local header sizes are 0).

using NpzFile = std::unordered_map<std::string, NpyArray>;

inline NpzFile load_npz(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
    throw std::runtime_error("load_npz: cannot open " + path);

  ifs.seekg(0, std::ios::end);
  size_t file_size = static_cast<size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  std::vector<char> fd(file_size);
  ifs.read(fd.data(), file_size);

  auto r16 = [&](size_t off) -> uint16_t {
    uint16_t v;
    std::memcpy(&v, fd.data() + off, 2);
    return v;
  };
  auto r32 = [&](size_t off) -> uint32_t {
    uint32_t v;
    std::memcpy(&v, fd.data() + off, 4);
    return v;
  };

  // Find End-of-Central-Directory record (last 22+ bytes)
  // Signature: PK\x05\x06
  size_t eocd_pos = file_size;
  {
    // Reverse search because the ZIP comment field can be 0..65535 bytes.
    size_t search_start = (file_size > 65557) ? file_size - 65557 : 0;
    for (size_t i = file_size - 22; i >= search_start; --i) {
      if (fd[i] == 'P' && fd[i + 1] == 'K' && fd[i + 2] == 0x05 &&
          fd[i + 3] == 0x06) {
        eocd_pos = i;
        break;
      }
      if (i == 0)
        break;
    }
  }
  if (eocd_pos == file_size)
    throw std::runtime_error("load_npz: no EOCD record in " + path);

  uint32_t cd_size = r32(eocd_pos + 12);
  uint32_t cd_offset = r32(eocd_pos + 16);

  // Walk central directory entries
  NpzFile npz;
  size_t cp = cd_offset;

  while (cp + 46 <= cd_offset + cd_size) {
    // Central directory file header: PK\x01\x02
    if (fd[cp] != 'P' || fd[cp + 1] != 'K' || fd[cp + 2] != 0x01 ||
        fd[cp + 3] != 0x02)
      break;

    uint16_t method = r16(cp + 10);
    uint32_t comp_size = r32(cp + 20);
    uint32_t uncomp_size = r32(cp + 24);
    uint16_t name_len = r16(cp + 28);
    uint16_t extra_len = r16(cp + 30);
    uint16_t comment_len = r16(cp + 32);
    uint32_t local_off = r32(cp + 42);

    std::string name(fd.data() + cp + 46, name_len);

    // Advance to next central directory entry
    cp += 46 + name_len + extra_len + comment_len;

    // Read data from local file header
    if (local_off + 30 > file_size)
      continue;
    uint16_t loc_name_len = r16(local_off + 26);
    uint16_t loc_extra_len = r16(local_off + 28);
    size_t data_pos = local_off + 30 + loc_name_len + loc_extra_len;

    if (data_pos + comp_size > file_size)
      throw std::runtime_error("load_npz: truncated entry: " + name);

    // Strip .npy extension for key
    std::string key = name;
    if (key.size() > 4 && key.substr(key.size() - 4) == ".npy")
      key = key.substr(0, key.size() - 4);

    std::vector<char> npy_buf;

    if (method == 0) {
      // Stored
      npy_buf.assign(fd.data() + data_pos, fd.data() + data_pos + comp_size);
    } else if (method == 8) {
      // Deflate
      npy_buf.resize(uncomp_size);
      z_stream zs{};
      if (inflateInit2(&zs, -MAX_WBITS) != Z_OK)
        throw std::runtime_error("load_npz: inflateInit2 failed");
      zs.next_in = reinterpret_cast<Bytef *>(fd.data() + data_pos);
      zs.avail_in = comp_size;
      zs.next_out = reinterpret_cast<Bytef *>(npy_buf.data());
      zs.avail_out = uncomp_size;
      int ret = inflate(&zs, Z_FINISH);
      inflateEnd(&zs);
      if (ret != Z_STREAM_END)
        throw std::runtime_error("load_npz: inflate failed for " + name);
    } else {
      throw std::runtime_error("load_npz: unsupported compression method " +
                               std::to_string(method) + " for " + name);
    }

    npz[key] = parse_npy(npy_buf.data(), npy_buf.size());
  }

  return npz;
}

// Convenience helpers.

inline bool has_key(const NpzFile &npz, const std::string &key) {
  return npz.find(key) != npz.end();
}

} // namespace npz
} // namespace jave
