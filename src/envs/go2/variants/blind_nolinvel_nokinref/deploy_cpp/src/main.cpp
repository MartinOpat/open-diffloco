/// @file main.cpp
/// Deploy JAVE policy on Unitree Go2.
///
/// Usage:
///   jave_deploy --policy policy_deploy.npz --interface lo --domain-id 1   #
///   sim jave_deploy --policy policy_deploy.npz --interface enp3s0 #

#include "controller.hpp"
#include "policy.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

static void print_usage(const char *prog) {
  std::cout << "Usage: " << prog << " --policy <path.npz> [options]\n\n"
            << "Options:\n"
            << "  --policy <path>       Exported .npz policy (required)\n"
            << "  --interface <iface>   Network interface (default: lo)\n"
            << "  --domain-id <id>      DDS domain ID (default: 0)\n"
            << "  --kp <value>          Override walk kp\n"
            << "  --kd <value>          Override walk kd\n"
            << "  --command-source <s>  terminal, wireless, or ros2 (default: terminal)\n"
            << "  --cmd-topic <topic>   ROS2 PointStamped cmd topic (default: "
               "/velocity_command)\n"
            << "\nExamples:\n"
            << "  " << prog
            << " --policy policy_deploy.npz --interface lo --domain-id 1\n"
            << "  " << prog
            << " --policy policy_deploy.npz --interface enp3s0\n";
}

int main(int argc, char **argv) {
  std::string policy_path;
  std::string interface = "lo";
  int domain_id = 0;
  double kp_override = -1, kd_override = -1;
  jave::CommandSource command_source = jave::CommandSource::TERMINAL;
  std::string cmd_topic = "/velocity_command";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--policy" || arg == "-p") && i + 1 < argc)
      policy_path = argv[++i];
    else if ((arg == "--interface" || arg == "-i") && i + 1 < argc)
      interface = argv[++i];
    else if ((arg == "--domain-id" || arg == "-d") && i + 1 < argc)
      domain_id = std::atoi(argv[++i]);
    else if (arg == "--kp" && i + 1 < argc)
      kp_override = std::atof(argv[++i]);
    else if (arg == "--kd" && i + 1 < argc)
      kd_override = std::atof(argv[++i]);
    else if (arg == "--command-source" && i + 1 < argc) {
      std::string value = argv[++i];
      if (value == "terminal")
        command_source = jave::CommandSource::TERMINAL;
      else if (value == "wireless")
        command_source = jave::CommandSource::WIRELESS;
      else if (value == "ros2")
        command_source = jave::CommandSource::ROS2;
      else {
        std::cerr << "Unknown command source: " << value << "\n";
        print_usage(argv[0]);
        return 1;
      }
    }
    else if (arg == "--cmd-topic" && i + 1 < argc)
      cmd_topic = argv[++i];
    else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  if (policy_path.empty()) {
    std::cerr << "Error: --policy is required.\n\n";
    print_usage(argv[0]);
    return 1;
  }

  auto policy = std::make_shared<jave::NumpyPolicy>(policy_path);
  jave::Go2Deploy controller(policy, interface, domain_id, command_source,
                             cmd_topic);

  if (kp_override > 0) {
    controller.kp = kp_override;
    std::cout << "  Override kp=" << kp_override << "\n";
  }
  if (kd_override > 0) {
    controller.kd = kd_override;
    std::cout << "  Override kd=" << kd_override << "\n";
  }

  controller.run();
  return 0;
}
