#pragma once

#include <string>
#include <utility>

namespace snapshot {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kOutOfMemory,
  kBackendUnavailable,
  kBackendError,
  kIoError,
  kFormatError,
  kCrcMismatch,
  kUnsupported,
  kMismatch,
  kNotFound,
  kOverflow,
};

class Status {
 public:
  Status() = default;
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  static Status Ok() { return {}; }
  static Status invalid_argument(std::string message) {
    return {StatusCode::kInvalidArgument, std::move(message)};
  }
  static Status backend(std::string message) {
    return {StatusCode::kBackendError, std::move(message)};
  }
  static Status unavailable(std::string message) {
    return {StatusCode::kBackendUnavailable, std::move(message)};
  }
  static Status unsupported(std::string message) {
    return {StatusCode::kUnsupported, std::move(message)};
  }
  static Status io(std::string message) {
    return {StatusCode::kIoError, std::move(message)};
  }
  static Status format(std::string message) {
    return {StatusCode::kFormatError, std::move(message)};
  }
  static Status mismatch(std::string message) {
    return {StatusCode::kMismatch, std::move(message)};
  }
  static Status overflow(std::string message) {
    return {StatusCode::kOverflow, std::move(message)};
  }

  bool ok() const { return code_ == StatusCode::kOk; }
  explicit operator bool() const { return ok(); }

  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

inline const char* status_code_name(StatusCode code) {
  switch (code) {
    case StatusCode::kOk:
      return "ok";
    case StatusCode::kInvalidArgument:
      return "invalid_argument";
    case StatusCode::kOutOfMemory:
      return "out_of_memory";
    case StatusCode::kBackendUnavailable:
      return "backend_unavailable";
    case StatusCode::kBackendError:
      return "backend_error";
    case StatusCode::kIoError:
      return "io_error";
    case StatusCode::kFormatError:
      return "format_error";
    case StatusCode::kCrcMismatch:
      return "crc_mismatch";
    case StatusCode::kUnsupported:
      return "unsupported";
    case StatusCode::kMismatch:
      return "mismatch";
    case StatusCode::kNotFound:
      return "not_found";
    case StatusCode::kOverflow:
      return "overflow";
  }
  return "unknown";
}

}  // namespace snapshot
