#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>  // automatically loads vulkan header along with GLFW's own definitions

#include <iostream>
#include <stdexcept>
#include <cstdlib>

#include <vector>
#include <cstring>

#include <map>
#include <optional> // from C++17


/// window parameters
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

/// config variable to the program to specify the layers to enable
const std::vector<const char*> validationLayers = {
  "VK_LAYER_KHRONOS_validation" // All of the useful standard validation is bundled into this layer included in the SDK
};

/// config variable to the program whether to enable the validation layer or not
/// NDEBUG -> not debug
#ifdef NDEBUG
  const bool enableValidationLayers = false;
#else
  const bool enableValidationLayers = true;
#endif


VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, 
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, 
                                      const VkAllocationCallbacks* pAllocator, 
                                      VkDebugUtilsMessengerEXT* pDebugMessenger) {
  // vkGetInstanceProcAddr will return nullptr if the function couldn't be loaded.
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if(func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  }
  else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}


void DestroyDebugUtilsMessengerEXT(VkInstance instance, 
                                  VkDebugUtilsMessengerEXT debugMessenger, 
                                  const VkAllocationCallbacks* pAllocator) {
  // This function should be either a static class function or a function outside the class.
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if(func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}


class TriangleApp {
  public:
    void run() {
      initWindow();
      initVulkan();
      mainLoop();
      cleanup();
    }

  private:
    GLFWwindow* window;
    VkInstance vkinstance;
    VkDebugUtilsMessengerEXT debugMessenger; // The debug callback in Vulkan is managed with a handle that needs to be explicitly created and destroyed.
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; // Implicitly destroyed, therefore need not do anything in cleanup.

    struct QueueFamilyIndices { // To hold info about different kinds of queue families supported by the physical device
      std::optional<uint32_t> graphicsFamily; 

      bool isComplete() {
        // At any point you can query if a std::optional<T> variable contains a value or not by calling its has_value() member function
        return graphicsFamily.has_value();
      }
    };


    /// Window creation 
    void initWindow() {
      glfwInit();  // init GLFW

      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // not create a window in a OpenGL context
      glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);    // disable window resizing

      window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }


    void initVulkan() {
      createInstance(); // initializes Vulkan library
      setupDebugMessenger(); // setup error handling
      pickPhysicalDevice();
    }


    /// Create an instance of the Vulkan library
    void createInstance() {
      // populate struct that holds info about our app
      VkApplicationInfo appInfo {};
      appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
      appInfo.pApplicationName = "Vulkan Triangle No. 1";
      appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);  // constructs an API version number.
      appInfo.pEngineName = "No Engine";
      appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
      appInfo.apiVersion = VK_API_VERSION_1_0;  // #define VK_API_VERSION_1_0 VK_MAKE_VERSION(1, 0, 0)

      // Tells the Vulkan driver which global extensions we want to use
      VkInstanceCreateInfo createInfo {};
      createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
      createInfo.pApplicationInfo = &appInfo;

      // Next 3 lines of code specify global extensions we want
      auto glfwExtensions = getRequiredExtensions();
      createInfo.enabledExtensionCount = static_cast<uint32_t>(glfwExtensions.size());
      createInfo.ppEnabledExtensionNames = glfwExtensions.data();

      std::cout << "***** Required extensions *****\n";
      for(int i = 0; i < glfwExtensions.size(); i++) {
        std::cout << "\t" << glfwExtensions[i] << "\n";
      }

      // call to check validation layers availability
      if(enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("Validation layers requested, but not available!");
      }

      // determine the global validation layers to enable and include
      VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
      if(enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
      }
      else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
      }

      VkResult result = vkCreateInstance(&createInfo, nullptr, &vkinstance); // return VK_SUCCESS or an error code
      if(result != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
      }

      /// NOTES
      /// Generic pattern to object creation parameters in function calls in vulkan
      /// 1 - Pointer to struct with creation info
      /// 2- Pointer to custom allocator callbacks, always nullptr in this tutorial
      /// 3 - Pointer to the variable that stores the handle to the new object

      // To retrieve a list of supported extensions
      uint32_t extensionCount = 0;
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
      std::vector<VkExtensionProperties> extensions(extensionCount);       // allocate an array to hold the extensions list
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data()); // query the extension details

      std::cout << "\n***** Available extensions *****\n";
      for(const auto& availableExt : extensions) {
        std::cout << '\t' << availableExt.extensionName;

        // check if it is in required list of extensions
        for(int i = 0; i < glfwExtensions.size(); i++) {
          if(strcmp(availableExt.extensionName, glfwExtensions[i]) == 0) {
            std::cout << " - Is required";
            break;
          }
        }
        std::cout << "\n";
      }
    }


    /// Checks if all of the requested validation layers are available
    bool checkValidationLayerSupport() {
      uint32_t layerCount;
      vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

      std::vector<VkLayerProperties> availableLayers(layerCount);
      vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

      // Check if all of the layers in validationLayers exist in the availableLayers list.
      for(const char* layerName : validationLayers) {
        bool layerFound = false;

        for(const auto& layerProperties : availableLayers) {
          if(strcmp(layerName, layerProperties.layerName) == 0) {
            layerFound = true;
            break;
          }
        }

        if(!layerFound) {
          return false;
        }
      }
      return true;
    }


    /// Returns the required list of extensions based on whether validation layers are enabled or not
    std::vector<const char*> getRequiredExtensions() {
      uint32_t glfwExtensionCount = 0;
      const char** glfwExtensions;

      // GLFW's built-in function that returns an array of the extension(s) it needs to interface with the window system
      // The extensions specified by GLFW are always required
      glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

      std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

      // The debug messenger extension is conditionally added
      if(enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
      }
      return extensions;
    }

    
    /// Debug callback function
    /// Returns a boolean that indicates if the Vulkan call that triggered the validation layer message should be aborted. 
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, // specifies the severity of the message
                                                        VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                        void* pUserData) {
      // pUserData: Contains a pointer that was specified during the setup of the callback and allows you to pass your own data to it.
      // pMessage: The debug message as a null-terminated string
      // pObjects: Array of Vulkan object handles related to the message
      // objectCount: Number of objects in array
      std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

      return VK_FALSE;
    }


    /// utility function to populate createInfo objects
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
      createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
      createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      createInfo.pfnUserCallback = debugCallback;
      //createInfo.pUserData = nullptr; // Optional
    }


    /// Tell Vulkan about the callback function we created above.
    void setupDebugMessenger() {
      if(!enableValidationLayers)
        return;

      VkDebugUtilsMessengerCreateInfoEXT createInfo {};
      populateDebugMessengerCreateInfo(createInfo);
      // This struct should be passed to the vkCreateDebugUtilsMessengerEXT function to create the VkDebugUtilsMessengerEXT object. 
      // This function is an extension function, it is not automatically loaded. 
      // We have to look up its address ourselves using vkGetInstanceProcAddr.
      // We're going to create our own proxy function that handles this in the background.

      // The debug messenger is specific to our Vulkan instance and its layers
      if(CreateDebugUtilsMessengerEXT(vkinstance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
      }
    }
    
    
    /// Look for and select a graphics card in the system that supports the features we need.
    void pickPhysicalDevice() {
      uint32_t deviceCount = 0;
      vkEnumeratePhysicalDevices(vkinstance, &deviceCount, nullptr);

      if(deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
      }

      // Allocate an array to hold all of the VkPhysicalDevice handles.
      std::vector<VkPhysicalDevice> devices(deviceCount);
      vkEnumeratePhysicalDevices(vkinstance, &deviceCount, devices.data());

      // Find a suitable device
      for(const auto& device : devices) {
        if(isDeviceSuitable(device)) {
          physicalDevice = device;
          break;
        }
      }
      if(physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
      }

      // Uncomment below block of code if using rateDeviceSuitability
      // // Begin uncomment
      //// Use an ordered map to automatically sort candidates by increasing score
      //std::multimap<int, VkPhysicalDevice> candidates;

      //for(const auto& device : devices) {
      //  int score = rateDeviceSuitability(device);
      //  candidates.insert(std::make_pair(score, device));
      //}

      //// Check if the best candidate is suitable at all
      //if(candidates.rbegin()->first > 0) {
      //  physicalDevice = candidates.rbegin()->second;
      //}
      //else {
      //  throw std::runtime_error("failed to find a suitable GPU!");
      //}
      // // End uncomment
    }


    /// Evaluate a physical device and check if it is suitable for the operations we want to perform
    bool isDeviceSuitable(VkPhysicalDevice device) {
      // Perform basic device suitability checks - name, type and supported vulkan version
      VkPhysicalDeviceProperties deviceProperties;
      vkGetPhysicalDeviceProperties(device, &deviceProperties);

      // Check support for optional features like texture compression, 64 bit floats and multi viewport rendering (useful for VR)
      VkPhysicalDeviceFeatures deviceFeatures;
      vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

      //// Would return next line if the application were more complex
      //return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && deviceFeatures.geometryShader;
      // But for now we just need Vulkan, so any GPU will do

      QueueFamilyIndices indices = findQueueFamilies(device);

      return indices.isComplete();
    }


    /// OPTIONAL - give each physical device a score and pick the most suitable one, but also have a fallback to an integrated GPU
    int rateDeviceSuitability(VkPhysicalDevice device) {
      int score = 0;

      // Perform basic device suitability checks - name, type and supported vulkan version
      VkPhysicalDeviceProperties deviceProperties;
      vkGetPhysicalDeviceProperties(device, &deviceProperties);

      // Check support for optional features like texture compression, 64 bit floats and multi viewport rendering (useful for VR)
      VkPhysicalDeviceFeatures deviceFeatures;
      vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

      // Discrete GPUs have a significant performance advantage
      if(deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 1000;
      }

      // Maximum possible size of textures affects graphics quality
      score += deviceProperties.limits.maxImageDimension2D;

      // Application can't function without geometry shaders
      if(!deviceFeatures.geometryShader) {
        return 0;
      }

      return score;
    }

    
    /// Check which queue families are supported by the device and which one of these supports the commands that we want to use
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
      QueueFamilyIndices indices;

      uint32_t queueFamilyCount = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

      // The VkQueueFamilyProperties struct contains some details about the queue family
      // including the type of operations that are supported and the number of queues that can be created based on that family
      std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
      vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
      // We need to find at least one queue family that supports VK_QUEUE_GRAPHICS_BIT
      int i = 0;
      for(const auto& queueFamily : queueFamilies) {
        if(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
          indices.graphicsFamily = i;
        }
        if(indices.isComplete()) { // break loop if we have found a suitable queue family
          break;
        }
        i++;
      }

      return indices;
    }


    /// Resource management
    void cleanup() {
      if(enableValidationLayers) {
        DestroyDebugUtilsMessengerEXT(vkinstance, debugMessenger, nullptr);
      }

      vkDestroyInstance(vkinstance, nullptr);
      glfwDestroyWindow(window);
      glfwTerminate();
    }


    void mainLoop() {
      while(!glfwWindowShouldClose(window)) {  // to keep the window open until it is closed or an error occurs
        glfwPollEvents();
      }
    }
};


int main() {
  // instance
  TriangleApp app;

  try {
    app.run();
  }
  catch (const std::exception& e) {
    // app failed to run
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE; // from cstdlib
  }

  return EXIT_SUCCESS; // from cstdlib
}


// // CODE BELOW CHECKS THAT ENVIRONMENT HAS BEEN SETUP CORRECTLY
/*
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>

int main() {
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);

  uint32_t extensionCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

  std::cout << extensionCount << " extensions supported\n";

  glm::mat4 matrix;
  glm::vec4 vec;
  auto test = matrix * vec;

  while(!glfwWindowShouldClose(window)) {
    glfwPollEvents();
  }

  glfwDestroyWindow(window);

  glfwTerminate();

}
*/