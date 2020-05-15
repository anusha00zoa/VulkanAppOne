#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>  // automatically loads vulkan header along with GLFW's own definitions

#include <iostream>
#include <stdexcept>
#include <cstdlib>

#include <vector>
#include <cstring>
#include <set>
#include <map>
#include <optional> // from C++17


// window parameters
const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;

// config variable to the program to specify the layers to enable
const std::vector<const char*> validationLayers = {
  "VK_LAYER_KHRONOS_validation" // All of the useful standard validation is bundled into this layer included in the SDK
};

// config variable to the program whether to enable the validation layer or not
// NDEBUG -> not debug
#ifdef NDEBUG
  const bool enableValidationLayers = false;
#else
  const bool enableValidationLayers = true;
#endif

#pragma region Validation-layers-utils
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
#pragma endregion


class TriangleApp {
  public:
    void run() {
      initWindow();
      initVulkan();
      mainLoop();
      cleanup();
    }

  private:
    #pragma region Class-members
    GLFWwindow*               window;
    VkInstance                vkinstance;
    VkDebugUtilsMessengerEXT  debugMessenger;                   // The debug callback in Vulkan is managed with a handle that needs to be explicitly created and destroyed.
    VkPhysicalDevice          physicalDevice  = VK_NULL_HANDLE; // Implicitly destroyed, therefore need not do anything in cleanup.
    VkDevice                  logicalDevice;                    // A logical device instance. You can create multiple logical devices from the same physical device if you have varying requirements.
    VkQueue                   graphicsQueue;                    // Store a handle to the graphics queue. 
                                                                // Device queues are implicitly cleaned up when the device is destroyed, so we don't need to do anything in cleanup.
    VkQueue                   presentQueue;                     // The presentation queue handle
    VkSurfaceKHR              surface;                          // Is platform agnostic, but its creation isn't because it depends on window system details
                                                                // It is destroyed before the application instance.

    struct QueueFamilyIndices {                                 // To hold info about different kinds of queue families supported by the physical device
      std::optional<uint32_t> graphicsFamily; 
      std::optional<uint32_t> presentFamily;

      bool isComplete() {
        // At any point you can query if a std::optional<T> variable contains a value or not by calling its has_value() member function
        return graphicsFamily.has_value() && presentFamily.has_value();
      }
    };
    #pragma endregion 


    #pragma region Application-instance
    /// Create an instance of the Vulkan library
    void createInstance() {
      // populate struct that holds info about our app
      VkApplicationInfo appInfo {};
      appInfo.sType               = VK_STRUCTURE_TYPE_APPLICATION_INFO;
      appInfo.pApplicationName    = "Vulkan Triangle No. 1";
      appInfo.applicationVersion  = VK_MAKE_VERSION(1, 0, 0); // constructs an API version number.
      appInfo.pEngineName         = "No Engine";
      appInfo.engineVersion       = VK_MAKE_VERSION(1, 0, 0);
      appInfo.apiVersion          = VK_API_VERSION_1_0;       // #define VK_API_VERSION_1_0 VK_MAKE_VERSION(1, 0, 0)

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
    #pragma endregion


    #pragma region Validation-layers
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
      createInfo.sType            = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
      createInfo.messageSeverity  = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      createInfo.messageType      = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      createInfo.pfnUserCallback  = debugCallback;
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
    #pragma endregion


    #pragma region Physical-devices-and-queue-families
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

      // Ensure that a device can present images to the surface we created.
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
      // And also look for a queue family that has the capability of presenting to our window surface
      int i = 0;
      for(const auto& queueFamily : queueFamilies) {
        if(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
          indices.graphicsFamily = i;
        }
        
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if(presentSupport) {
          indices.presentFamily = i;
        }
        
        if(indices.isComplete()) { // break loop if we have found a suitable queue family
          break;
        }
        i++;
      }
      return indices;
    }
    #pragma endregion


    #pragma region Logical-devices-and-queues
    /// Set up a logical device to interface with the selected physical device
    void createLogicalDevice() {
      QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

      // VkDeviceQueueCreateInfo is a structure that describes the number of queues we want for a single queue family.
      // We need to have multiple VkDeviceQueueCreateInfo structs to create a queue from both families. 
      // An elegant way to do that is to create a set of all unique queue families that are necessary for the required queues
      std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
      std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

      /// NOTES
      /// Vulkan lets you assign priorities to queues to influence the scheduling of command buffer execution 
      /// using floating point numbers between 0.0 and 1.0. 
      /// This is required even if there is only a single queue.s
      float queuePriority = 1.0f;
      for(uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
      }

      // Specify the set of device features that we'll be using
      VkPhysicalDeviceFeatures deviceFeatures {};

      // Begin creating the logical device
      VkDeviceCreateInfo createInfo {};
      createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      
      // Add pointers to the queues creation info and device features structs
      createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
      createInfo.pQueueCreateInfos = queueCreateInfos.data();
      createInfo.pEnabledFeatures = &deviceFeatures;
      
      // Specify extensions (these are device specific)
      createInfo.enabledExtensionCount = 0;
      
      // Specify validation layers (these are device specific)
      if(enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
      }
      else {
        createInfo.enabledLayerCount = 0;
      }

      // Instantiate the logical device
      // This call can return errors based on enabling non-existent extensions or specifying the desired usage of unsupported features.
      // This device should be destroyed during cleanup
      if(vkCreateDevice(physicalDevice,                   // the physical device to interface with  
                        &createInfo,                      // the queue and usage info we just specified
                        nullptr,                          // the optional allocation callbacks pointer
                        &logicalDevice) != VK_SUCCESS) {  // a pointer to a variable to store the logical device handle in
        throw std::runtime_error("failed to create logical device!");
      }

      // Retrieve a queue handle for the graphics queue family.
      vkGetDeviceQueue(logicalDevice, indices.graphicsFamily.value(), 0, &graphicsQueue);
      vkGetDeviceQueue(logicalDevice, indices.presentFamily.value(), 0, &presentQueue);
    }
    #pragma endregion


    #pragma region Window-surface
    /// NOTES
    /// To establish the connection between Vulkan and the window system to present results to the screen, 
    /// we need to use the WSI (Window System Integration) extensions
    /// The surface in our program will be backed by the window that we've already opened with GLFW.
    /// The window surface needs to be created right after the instance creation, 
    /// because it can actually influence the physical device selection.
    /// Window surfaces are an entirely optional component in Vulkan, if you just need off-screen rendering. 
    /// Vulkan allows you to do that without hacks like creating an invisible window (necessary for OpenGL)

    /// Using the platform specific extension 'VK_KHR_win32_surface' to create a surface - not needed for this tutorial
    /// VkWin32SurfaceCreateInfoKHR createInfo {};
    /// createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    /// createInfo.hwnd = glfwGetWin32Window(window); // used to get the raw HWND from the GLFW window object
    /// createInfo.hinstance = GetModuleHandle(nullptr); // returns the HINSTANCE handle of the current process.
    /// if(vkCreateWin32SurfaceKHR(vkinstance, &createInfo, nullptr, &surface) != VK_SUCCESS) { 
    ///   throw std::runtime_error("failed to create window surface!");
    /// }
    /// The 'glfwCreateWindowSurface' function performs exactly this operation with a different implementation for each platform.

    void createSurface() {
      if(glfwCreateWindowSurface(vkinstance,                  // the VkInstance
                                  window,                     // GLFW window pointer 
                                  nullptr,                    // custom allocators 
                                  &surface) != VK_SUCCESS) {  // pointer to VkSurfaceKHR variable
        throw std::runtime_error("failed to create window surface!");
      }
    }  
    #pragma endregion


    #pragma region Base-code
    /// Window creation 
    void initWindow() {
      glfwInit();  // init GLFW

      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // not create a window in a OpenGL context
      glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);    // disable window resizing

      window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }


    void initVulkan() {
      // ORDER OF FUNCTION CALLS SHOULD NOT BE CHANGED
      createInstance();       // initializes Vulkan library
      setupDebugMessenger();  // setup error handling
      createSurface();        // creates window surface
      pickPhysicalDevice();   // selects a suitable physical device
      createLogicalDevice();  // creates a logical device to interface with the selected physical device
    }


    /// Resource management
    void cleanup() {
      vkDestroyDevice(logicalDevice, nullptr);

      if(enableValidationLayers) {
        DestroyDebugUtilsMessengerEXT(vkinstance, debugMessenger, nullptr);
      }

      vkDestroySurfaceKHR(vkinstance, surface, nullptr);
      vkDestroyInstance(vkinstance, nullptr);
      glfwDestroyWindow(window);
      glfwTerminate();
    }


    void mainLoop() {
      while(!glfwWindowShouldClose(window)) {  // to keep the window open until it is closed or an error occurs
        glfwPollEvents();
      }
    }
    #pragma endregion
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