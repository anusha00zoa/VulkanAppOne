#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>  // automatically loads vulkan header along with GLFW's own definitions

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdlib>

#include <vector>
#include <cstring>
#include <cstdint> // Necessary for UINT32_MAX
#include <set>
#include <map>
#include <algorithm>
#include <optional> // from C++17


// window parameters
const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;

// config variable to the program to specify the layers to enable
const std::vector<const char*> validationLayers = {
  // All of the useful standard validation is bundled into this layer included in the SDK
  "VK_LAYER_KHRONOS_validation"
};

// config variable to the program to specify the list of required device extensions
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
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
    // The debug callback handle that needs to be explicitly created and destroyed.
    VkDebugUtilsMessengerEXT  debugMessenger;  
    // Implicitly destroyed, therefore need not do anything in cleanup.
    VkPhysicalDevice          physicalDevice  = VK_NULL_HANDLE;
    // A logical device instance. You can create multiple logical devices from the same physical 
    // device if you have varying requirements.
    VkDevice                  logicalDevice;                    

    // Should be destroyed before the logical device
    VkSwapchainKHR            swapChain;
    // To store the handles of the 'VKImage's that will be in the swap chain
    std::vector<VkImage>      swapChainImages;
    // Store the swapchain's surface format
    VkFormat                  swapChainImageFormat;
    // Store the swapchain's swap extent
    VkExtent2D                swapChainExtent;
    // Store the image views.
    std::vector<VkImageView>  swapChainImageViews;

    // Device queues are implicitly cleaned up when the device is destroyed.
    // Store a handle to the graphics queue. 
    VkQueue                   graphicsQueue;
    // The presentation queue handle
    VkQueue                   presentQueue;   
    // VkSurfaceKHR is platform agnostic, but its creation isn't because it depends on window 
    // system details. It is destroyed before the application instance.
    VkSurfaceKHR              surface;
    VkPipelineLayout          pipelineLayout;
    VkRenderPass              renderPass;
    VkPipeline                graphicsPipeline;
    
    // To hold info about different kinds of queue families supported by the physical device
    struct QueueFamilyIndices {                                 
      std::optional<uint32_t> graphicsFamily; 
      std::optional<uint32_t> presentFamily;

      bool isComplete() {
        // At any point you can query if a std::optional<T> variable contains a value or not 
        // by calling its has_value() member function
        return graphicsFamily.has_value() && presentFamily.has_value();
      }
    };

    /// NOTES
    /// Simply checking if a swap chain is available is not sufficient, because it may not actually
    /// be compatible with our window surface. Creating a swap chain also involves a lot more
    /// settings than instance and device creation, so we need to query for some more details 
    /// before we're able to proceed.  There are basically 3 kinds of properties we need to check:
    /// 1. Basic surface capabilities (min / max number of images in swap chain, 
    ///     min / max width and height of images)
    /// 2. Surface formats(pixel format, color space)
    /// 3. Available presentation modes
    struct SwapChainSupportDetails {
      VkSurfaceCapabilitiesKHR capabilities;
      std::vector<VkSurfaceFormatKHR> formats;
      std::vector<VkPresentModeKHR> presentModes;
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
      appInfo.apiVersion          = VK_API_VERSION_1_0;
      // #define VK_API_VERSION_1_0 VK_MAKE_VERSION(1, 0, 0)

      // Tells the Vulkan driver which global extensions we want to use
      VkInstanceCreateInfo createInfo {};
      createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
      createInfo.pApplicationInfo = &appInfo;

      // specify global extensions we want
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

      VkResult result = vkCreateInstance(&createInfo, nullptr, &vkinstance);
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
      // allocate an array to hold the extensions list
      std::vector<VkExtensionProperties> extensions(extensionCount);
      // query the extension details
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data()); 

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
    /// Returns a boolean that indicates if the Vulkan call that triggered the validation 
    /// layer message should be aborted. 
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData) {
      // pUserData: Contains a pointer that was specified during the setup of the callback 
      //              and allows you to pass your own data to it.
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
      createInfo.messageSeverity  = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT 
                                      | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT 
                                      | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      createInfo.messageType      = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT 
                                    | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT 
                                    | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      createInfo.pfnUserCallback  = debugCallback;
      //createInfo.pUserData = nullptr; // Optional
    }


    /// Tell Vulkan about the callback function we created above.
    void setupDebugMessenger() {
      if(!enableValidationLayers)
        return;

      VkDebugUtilsMessengerCreateInfoEXT createInfo {};
      populateDebugMessengerCreateInfo(createInfo);
      // This struct should be passed to the vkCreateDebugUtilsMessengerEXT function to create the
      // VkDebugUtilsMessengerEXT object. This function is an extension function, it is not 
      // automatically loaded. We have to look up its address using vkGetInstanceProcAddr. 
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

      // // Check if the best candidate is suitable at all
      //if(candidates.rbegin()->first > 0) {
      //  physicalDevice = candidates.rbegin()->second;
      //}
      //else {
      //  throw std::runtime_error("failed to find a suitable GPU!");
      //}
      // // End uncomment
    }


    /// Evaluate a physical device, check if it is suitable for the operations we want to perform
    /// i.e. The device can present images to the surface we created.
    bool isDeviceSuitable(VkPhysicalDevice device) {
      // Perform basic device suitability checks - name, type and supported vulkan version
      VkPhysicalDeviceProperties deviceProperties;
      vkGetPhysicalDeviceProperties(device, &deviceProperties);

      // Check support for optional features - texture compression, 64 bit floats 
      // and multi viewport rendering (useful for VR)
      VkPhysicalDeviceFeatures deviceFeatures;
      vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

      // Would return next line if the application were more complex
      // But for now we just need Vulkan, so any GPU will do
      //return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU 
      //                                        && deviceFeatures.geometryShader;

      QueueFamilyIndices indices = findQueueFamilies(device);

      bool extensionsSupported = checkDeviceExtensionSupport(device);

      // Verify that swap chain support is adequate, after verifying that extension is available
      bool swapChainAdequate = false;
      if(extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
      }

      return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }


    /// OPTIONAL - give each physical device a score and pick the most suitable one, 
    /// but also have a fallback to an integrated GPU
    int rateDeviceSuitability(VkPhysicalDevice device) {
      int score = 0;

      // Perform basic device suitability checks - name, type and supported vulkan version
      VkPhysicalDeviceProperties deviceProperties;
      vkGetPhysicalDeviceProperties(device, &deviceProperties);

      // Check support for optional features like texture compression, 
      // 64 bit floats and multi viewport rendering (useful for VR)
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

    
    /// Check which queue families are supported by the device and 
    /// which one of these supports the commands that we want to use
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
      QueueFamilyIndices indices;

      uint32_t queueFamilyCount = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

      // The VkQueueFamilyProperties struct contains some details about the queue family
      // including the type of operations that are supported and 
      // the number of queues that can be created based on that family
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

      // VkDeviceQueueCreateInfo is a structure that describes the number of queues we want for a
      // single queue family. We need to have multiple VkDeviceQueueCreateInfo structs to create a
      // queue from both families. An elegant way to do that is to create a set of all unique queue
      // families that are necessary for the required queues
      std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
      std::set<uint32_t> uniqueQueueFamilies = {  
        indices.graphicsFamily.value(), 
        indices.presentFamily.value()
      };

      /// NOTES
      /// Vulkan lets you assign priorities to queues to influence the scheduling of command buffer
      /// execution using floating point numbers between 0.0 and 1.0. 
      /// This is required even if there is only a single queue.
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
      createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
      createInfo.ppEnabledExtensionNames = deviceExtensions.data();
      
      // Specify validation layers (these are device specific)
      if(enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
      }
      else {
        createInfo.enabledLayerCount = 0;
      }

      // Instantiate the logical device
      // This call can return errors based on enabling non-existent extensions or specifying the
      // desired usage of unsupported features. This device should be destroyed during cleanup
      if(vkCreateDevice(physicalDevice, // the physical device to interface with  
                        &createInfo,    // the queue and usage info we just specified
                        nullptr,        // the optional allocation callbacks pointer
                        &logicalDevice) // pointer to a variable to store the logical device handle
                        != VK_SUCCESS) {  
        throw std::runtime_error("failed to create logical device!");
      }

      // Retrieve a queue handle for the graphics queue family.
      vkGetDeviceQueue(logicalDevice, indices.graphicsFamily.value(), 0, &graphicsQueue);
      vkGetDeviceQueue(logicalDevice, indices.presentFamily.value(), 0, &presentQueue);
    }
    #pragma endregion


    #pragma region Window-surface
    /// NOTES
    /// To establish the connection between Vulkan and the window system to present results to the
    /// screen, we need to use the WSI (Window System Integration) extensions. The surface in our
    /// program will be backed by the window that we've already opened with GLFW. The window 
    /// surface needs to be created right after the instance creation, because it can actually 
    /// influence the physical device selection. Window surfaces are an entirely optional component
    /// in Vulkan, if you just need off-screen rendering. Vulkan allows you to do that without 
    /// hacks like creating an invisible window (necessary for OpenGL)

    /// Using the platform specific extension 'VK_KHR_win32_surface' to create a surface
    /// NOT needed for this tutorial
    /// 
    /// VkWin32SurfaceCreateInfoKHR createInfo {};
    /// createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    /// createInfo.hwnd = glfwGetWin32Window(window); // get raw HWND from the GLFW window object
    /// createInfo.hinstance = GetModuleHandle(nullptr); // returns handle of current process
    /// if(vkCreateWin32SurfaceKHR(vkinstance, &createInfo, nullptr, &surface) != VK_SUCCESS) { 
    ///   throw std::runtime_error("failed to create window surface!");
    /// }
    /// 
    /// The 'glfwCreateWindowSurface' function performs exactly this operation with a different
    /// implementation for each platform.

    void createSurface() {
      if(glfwCreateWindowSurface(vkinstance,                  // the VkInstance
                                  window,                     // GLFW window pointer 
                                  nullptr,                    // custom allocators 
                                  &surface) != VK_SUCCESS) {  // pointer to VkSurfaceKHR variable
        throw std::runtime_error("failed to create window surface!");
      }
    }  
    #pragma endregion


    #pragma region Swapchain
    /// Function to enumerate the extensions and check if all required extensions are amongst them.
    /// Using a swapchain requires enabling the VK_KHR_swapchain extension.
    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
      uint32_t extensionCount;
      vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

      std::vector<VkExtensionProperties> availableExtensions(extensionCount);
      vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

      std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

      for(const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
      }

      return requiredExtensions.empty();
    }

    /// Function to populate our swapchain struct
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
      SwapChainSupportDetails details;

      // Query basic surface capabilities
      // The VkPhysicalDevice and VkSurfaceKHR window surface are the first two parameters 
      // because they are the core components of the swap chain.
      vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

      // Querying the supported surface formats
      uint32_t formatCount;
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
      if(formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
      }

      // Querying the supported presentation modes
      uint32_t presentModeCount;
      vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
      if(presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
      }

      return details;
    }

    /// NOTES
    /// Choosing the right settings for the swap chain
    /// There are three types of settings to determine:
    /// 1. Surface format(color depth)
    /// 2. Presentation mode(conditions for "swapping" images to the screen)
    /// 3. Swap extent(resolution of images in swap chain)

    /// Choose surface format
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
      // Each VkSurfaceFormatKHR entry contains a format and a colorSpace member. 
      // The format member specifies the color channels and types.
      // The colorSpace member indicates if the SRGB color space is supported or not
      for(const auto& availableFormat : availableFormats) {
        if(availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB 
            && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
          return availableFormat;
        }
      }
      return availableFormats[0];
    }

    /// Choose presentation mode
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
      // If mailbox mode is available, choose that
      for(const auto& availablePresentMode : availablePresentModes) {
        if(availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
          return availablePresentMode;
        }
      }
      
      // Else choose fifo mode
      return VK_PRESENT_MODE_FIFO_KHR;
    }

    /// Choose swap extent - the resolution of the swap chain images
    /// Almost always exactly equal to the resolution of the window that we're drawing to
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
      // Match resolution of the window by setting width and height in the currentExtent member.
      if(capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
      }
      else {
        VkExtent2D actualExtent = {WIDTH, HEIGHT};

        actualExtent.width = std::max(capabilities.minImageExtent.width, 
                                      std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, 
                                      std::min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
      }
    }

    /// Function to create the actual swap chain
    void createSwapChain() {
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

      VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
      VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
      VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

      // Decide how many images we would like to have in the swap chain, make sure to not exceed 
      // the maximum number of images while doing this. Here 0 means that there is no maximum.
      uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
      if(swapChainSupport.capabilities.maxImageCount > 0 
          && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
      }

      VkSwapchainCreateInfoKHR createInfo {};
      createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
      createInfo.surface = surface;
      createInfo.minImageCount = imageCount;
      createInfo.imageFormat = surfaceFormat.format;
      createInfo.imageColorSpace = surfaceFormat.colorSpace;
      createInfo.imageExtent = extent;
      // specifies the amount of layers each image consists of. 
      // This is always 1 unless you are developing a stereoscopic 3D application.
      createInfo.imageArrayLayers = 1;
      // specifies what kind of operations we'll use the images in the swap chain for.
      createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

      QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
      uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
      if(indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
      }
      else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0; // Optional
        createInfo.pQueueFamilyIndices = nullptr; // Optional
      }

      /// NOTES
      /// We need to specify how to handle swap chain images that will be used across multiple 
      /// queue families. That will be the case in our application if the graphics queue family is
      /// different from the presentation queue. If the queue families differ, then we'll be using
      /// the concurrent mode in this tutorial. Concurrent mode requires you to specify in advance
      /// between which queue families ownership will be shared using the queueFamilyIndexCount and
      /// pQueueFamilyIndices parameters. If the graphics queue family and presentation queue 
      /// family are the same, which will be the case on most hardware, then we should stick to 
      /// exclusive mode, because concurrent mode requires you to specify at least two distinct
      /// queue families.
      
      // To specify that you do not want any transformation on images
      createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
      // specifies if alpha channel should be used for blending with other windows in the system
      createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; 
      createInfo.presentMode = presentMode;
      createInfo.clipped = VK_TRUE;

      /// NOTES
      /// it's possible that your swap chain becomes invalid or unoptimized while your application
      /// is running, for eg. because the window was resized. In that case, the swap chain 
      /// needs to be recreated from scratch and a reference to the old one must be specified. 
      createInfo.oldSwapchain = VK_NULL_HANDLE;

      if(vkCreateSwapchainKHR(logicalDevice, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
      }

      // Retrieving the swapchain images
      vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, nullptr);
      swapChainImages.resize(imageCount);
      vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, swapChainImages.data());

      // Assign to member variables
      swapChainImageFormat = surfaceFormat.format;
      swapChainExtent = extent;
    }
    #pragma endregion


    #pragma region Image-views
    /// NOTES
    /// To use any 'VkImage', including those in the swap chain, in the render pipeline we have to 
    /// create a 'VkImageView' object. An image view is quite literally a view into an image. It 
    /// describes how to access the image and which part of the image to access, for example if it 
    /// should be treated as a 2D texture depth texture without any mipmapping levels.

    /// A function that creates a basic image view for every image in the swap chain 
    /// so that we can use them as color targets
    void createImageViews() {
      // resize the list to fit all of the image views we'll be creating
      swapChainImageViews.resize(swapChainImages.size());

      // iterate over all of the swap chain images.
      for(size_t i = 0; i < swapChainImages.size(); i++) {
        VkImageViewCreateInfo createInfo {};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapChainImages[i];
        
        // specify how the image data should be interpreted
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapChainImageFormat;

        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        
        // describes what the image's purpose is and which part of the image should be accessed
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        /// NOTES
        /// If you were working on a stereographic 3D app, then you would create a swapchain with
        /// multiple layers. You could then create multiple image views for each image representing
        /// the views for the left and right eyes by accessing different layers.

        if(vkCreateImageView(logicalDevice, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
          throw std::runtime_error("failed to create image views!");
        }

        /// add a similar loop to destroy the image views created now at the end of the program
      }
    }
    #pragma endregion


    #pragma region Graphics-pipeline
    /// A simple helper function to load the binary data from the compiled shader files
    static std::vector<char> readFile(const std::string& filename) {
      std::ifstream file(filename, std::ios::ate | std::ios::binary);

      if(!file.is_open()) {
        throw std::runtime_error("failed to open file!");
      }

      size_t fileSize = (size_t)file.tellg(); // get size of file
      std::vector<char> buffer(fileSize);

      file.seekg(0);
      // read all of the bytes from the specified file
      file.read(buffer.data(), fileSize);
      file.close();

      // return them in a byte array
      return buffer;
    }


    /// A helper function to create shader modules
    /// Shader modules are just a thin wrapper around the shader bytecode 
    VkShaderModule createShaderModule(const std::vector<char>& code) {
      VkShaderModuleCreateInfo createInfo {};
      createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      createInfo.codeSize = code.size();
      createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

      VkShaderModule shaderModule;
      if(vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
      }

      return shaderModule;
    }


    void createGraphicsPipeline() {
      // Load the bytecode of the two. Make sure that the shaders are loaded correctly - print the
      //  size of the buffers and check if they match the actual file size in bytes.
      auto vertShaderCode = readFile("shaders/vert.spv");
      auto fragShaderCode = readFile("shaders/frag.spv");
      
      // initialize the shader module variables
      VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
      VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

      // To actually use the shaders we'll need to assign them to a specific pipeline stage through
      //  VkPipelineShaderStageCreateInfo structures 
      // Now, fill in the structure for the vertex shader
      VkPipelineShaderStageCreateInfo vertShaderStageInfo {};
      vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      // telling Vulkan in which pipeline stage the shader is going to be used
      vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
      vertShaderStageInfo.module = vertShaderModule; // the shader module containing the code
      vertShaderStageInfo.pName = "main"; // the 'entrypoint' into the code
      
      // Now a similar structure for the fragment shader
      VkPipelineShaderStageCreateInfo fragShaderStageInfo {};
      fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
      fragShaderStageInfo.module = fragShaderModule;
      fragShaderStageInfo.pName = "main";

      VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

      // Vertex input
      // describes the format of the vertex data that will be passed to the vertex shader in 2 ways
      //  1. Bindings: spacing between data and whether the data is per-vertex or per-instance
      //  2. Attribute descriptions: type of the attributes passed to the vertex shader, 
      //      which binding to load them from and at which offset
      VkPipelineVertexInputStateCreateInfo vertexInputInfo {};
      vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
      vertexInputInfo.vertexBindingDescriptionCount = 0;
      vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
      vertexInputInfo.vertexAttributeDescriptionCount = 0;
      vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional

      // Input assembly
      // Describes two things: what kind of geometry will be drawn from the vertices and 
      //  if primitive restart should be enabled.
      VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
      inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
      inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      inputAssembly.primitiveRestartEnable = VK_FALSE;

      // Viewports and scissors
      // A viewport describes the region of the framebuffer that the output will be rendered to
      VkViewport viewport {};
      viewport.x = 0.0f;
      viewport.y = 0.0f;
      viewport.width = (float)swapChainExtent.width;
      viewport.height = (float)swapChainExtent.height;
      viewport.minDepth = 0.0f;
      viewport.maxDepth = 1.0f;
      // Scissor rectangles define in which regions pixels will actually be stored. Any pixels
      //  outside the scissor rectangles will be discarded by the rasterizer. 
      // They function like a filter rather than a transformation.
      VkRect2D scissor {};
      scissor.offset = {0, 0};
      scissor.extent = swapChainExtent;
      // Now this viewport and scissor rectangle need to be combined into a viewport state
      VkPipelineViewportStateCreateInfo viewportState {};
      viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
      viewportState.viewportCount = 1;
      viewportState.pViewports = &viewport;
      viewportState.scissorCount = 1;
      viewportState.pScissors = &scissor;

      // Rasterizer
      // Takes the geometry that is shaped by the vertices from the vertex shader and turns it into
      // fragments to be colored by the fragment shader. It also performs depth testing, face 
      // culling and the scissor test, and it can be configured to output fragments that fill 
      // entire polygons or just the edges(wireframe rendering)
      // If 'depthClampEnable' is set to VK_TRUE, then fragments that are beyond the near and far
      //  planes are clamped to them as opposed to discarding them
      // If 'rasterizerDiscardEnable' is set to VK_TRUE, then geometry never passes through the
      //  rasterizer stage.
      // 'polygonMode' determines how fragments are generated for geometry:
      //    1. VK_POLYGON_MODE_FILL: fill the area of the polygon with fragments
      //    2. VK_POLYGON_MODE_LINE: polygon edges are drawn as lines
      //    3. VK_POLYGON_MODE_POINT: polygon vertices are drawn as points
      // 'lineWidth' describes the thickness of lines in terms of number of fragments
      // The 'cullMode' variable determines the type of face culling to use. You can disable 
      //  culling, cull the front faces, cull the back faces or both. 
      // The 'frontFace' variable specifies the vertex order for faces to be considered 
      //  front-facing and can be clockwise or counterclockwise.
      VkPipelineRasterizationStateCreateInfo rasterizer {};
      rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
      rasterizer.depthClampEnable = VK_FALSE;
      rasterizer.rasterizerDiscardEnable = VK_FALSE;
      rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
      rasterizer.lineWidth = 1.0f;
      rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
      rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
      rasterizer.depthBiasEnable = VK_FALSE;
      rasterizer.depthBiasConstantFactor = 0.0f; // Optional
      rasterizer.depthBiasClamp = 0.0f; // Optional
      rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

      // Multisampling
      // One of the ways to perform anti-aliasing. It works by combining the fragment shader's
      // results of multiple polygons that rasterize to the same pixel. This mainly occurs along
      // edges. It doesn't need to run the fragment shader multiple times if only one polygon maps
      // to a pixel, it is significantly less expensive than simply rendering to a higher 
      // resolution and then downscaling.
      VkPipelineMultisampleStateCreateInfo multisampling {};
      multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
      multisampling.sampleShadingEnable = VK_FALSE;
      multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      multisampling.minSampleShading = 1.0f; // Optional
      multisampling.pSampleMask = nullptr; // Optional
      multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
      multisampling.alphaToOneEnable = VK_FALSE; // Optional

      // Depth and stencil testing
      // not required right now, so pass a nullptr

      // Color blending
      // After a fragment shader has returned a color, it needs to be combined with the color that
      // is already in the framebuffer.This transformation is known as color blending and there are
      // two ways to do it: Mix the old and new value to produce a final color [OR] Combine the old
      // and new value using a bitwise operation.
      // VkPipelineColorBlendAttachmentState contains the configuration per attached framebuffer 
      // If 'blendEnable' = VK_FALSE, the color from fragment shader is passed through unmodified
      VkPipelineColorBlendAttachmentState colorBlendAttachment {};
      colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      colorBlendAttachment.blendEnable = VK_FALSE;
      colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
      colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
      colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
      colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
      colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
      colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional
      // VkPipelineColorBlendStateCreateInfo contains the global color blending settings
      // This structure references the array of structures for all of the framebuffers and allows
      // you to set blend constants that you can use as blend factors
      VkPipelineColorBlendStateCreateInfo colorBlending {};
      colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
      colorBlending.logicOpEnable = VK_FALSE;
      colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
      colorBlending.attachmentCount = 1;
      colorBlending.pAttachments = &colorBlendAttachment;
      colorBlending.blendConstants[0] = 0.0f; // Optional
      colorBlending.blendConstants[1] = 0.0f; // Optional
      colorBlending.blendConstants[2] = 0.0f; // Optional
      colorBlending.blendConstants[3] = 0.0f; // Optional

      // Dynamic state
      // A limited amount of the state that we've specified in the previous structs can actually be
      // changed without recreating the pipeline. Examples are the size of the viewport, line width
      // and blend constants.
      VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_LINE_WIDTH
      };
      VkPipelineDynamicStateCreateInfo dynamicState {};
      dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
      dynamicState.dynamicStateCount = 2;
      dynamicState.pDynamicStates = dynamicStates;

      // Pipeline layout
      // If you use uniform values in shaders, these values need to be specified during pipeline
      // creation by creating a VkPipelineLayout object.
      // 'push constants' are another way of passing dynamic values to shaders
      VkPipelineLayoutCreateInfo pipelineLayoutInfo {};
      pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipelineLayoutInfo.setLayoutCount = 0; // Optional
      pipelineLayoutInfo.pSetLayouts = nullptr; // Optional
      pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
      pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

      if(vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
      }

      // Create the graphics pipeline using everything created above
      // Vulkan allows you to create a new graphics pipeline by deriving from an existing pipeline.
      // The idea of pipeline derivatives is that it is less expensive to set up pipelines when
      // they have much functionality in common with an existing pipeline and switching between
      // pipelines from the same parent can also be done quicker. You can either specify the handle
      // of an existing pipeline with 'basePipelineHandle' or reference another pipeline that is
      // about to be created by index with 'basePipelineIndex'.
      VkGraphicsPipelineCreateInfo pipelineInfo {};
      pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
      pipelineInfo.stageCount = 2;
      pipelineInfo.pStages = shaderStages;
      pipelineInfo.pVertexInputState = &vertexInputInfo;
      pipelineInfo.pInputAssemblyState = &inputAssembly;
      pipelineInfo.pViewportState = &viewportState;
      pipelineInfo.pRasterizationState = &rasterizer;
      pipelineInfo.pMultisampleState = &multisampling;
      pipelineInfo.pDepthStencilState = nullptr; // Optional
      pipelineInfo.pColorBlendState = &colorBlending;
      pipelineInfo.pDynamicState = nullptr; // Optional
      pipelineInfo.layout = pipelineLayout;
      pipelineInfo.renderPass = renderPass;
      pipelineInfo.subpass = 0;
      pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
      pipelineInfo.basePipelineIndex = -1; // Optional

      if(vkCreateGraphicsPipelines(logicalDevice, 
                                  VK_NULL_HANDLE, // an optional VkPipelineCache object. 
                                  1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
      }

      // cleanup of the shader code variables
      vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
      vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);
    }

    /// NOTES on shaders
    /// Shader code in Vulkan has to be specified in a bytecode format as opposed to human-readable
    /// syntax like GLSL and HLSL. This bytecode format is called SPIR-V and is designed to be used
    /// with both Vulkan and OpenCL (both Khronos APIs). The advantage of using a bytecode format
    /// is that the compilers written by GPU vendors to turn shader code into native code are
    /// significantly less complex. The past has shown that with human-readable syntax like GLSL,
    /// some GPU vendors were rather flexible with their interpretation of the standard.
    ///
    /// GLSL is a shading language with a C-style syntax. Programs written in it have a main
    /// function that is invoked for every object. Instead of using parameters for input and a 
    /// return value as output, GLSL uses global variables to handle input and output.
    ///
    /// The vertex shader processes each incoming vertex. It takes its attributes, like world 
    /// position, color, normal and texture coordinates as input. The output is the final position
    /// in clip coordinates and the attributes that need to be passed on to the fragment shader,
    /// like color and texture coordinates. These values will then be interpolated over the 
    /// fragments by the rasterizer to produce a smooth gradient.
    ///
    /// A 'clip coordinate' is a 4D vector from the vertex shader that is subsequently turned into
    /// a normalized device coordinate by dividing the whole vector by its last component. These
    /// normalized device coordinates are homogeneous coordinates that map the framebuffer to 
    /// a [-1, 1] by [-1, 1] coordinate system.

    /// NOTES on Render Passes
    /// We need to tell Vulkan about the framebuffer attachments that will be used while rendering.
    /// We need to specify how many color and depth buffers there will be, how many samples to use
    /// for each of them and how their contents should be handled throughout the rendering
    /// operations. All of this information is wrapped in a render pass object.
    ///
    /// A single render pass can consist of multiple subpasses. Subpasses are subsequent rendering
    /// operations that depend on the contents of framebuffers in previous passes, for example a 
    /// sequence of post-processing effects that are applied one after another. If you group these
    /// rendering operations into one render pass, then Vulkan is able to reorder the operations 
    /// and conserve memory bandwidth for possibly better performance. Every subpass references one
    /// or more of the attachments that we've described using the structure. 

    void createRenderPass() {
      // A single color buffer attachment represented by one of the images from the swap chain.
      // The 'format' of the color attachment should match the format of the swap chain images
      // The 'loadOp' and 'storeOp' determine what to do with the data in the attachment before
      // rendering and after rendering.
      // The 'initialLayout' specifies which layout the image will have before the render pass 
      // begins. The finalLayout specifies the layout to automatically transition to when the 
      // render pass finishes. Using VK_IMAGE_LAYOUT_UNDEFINED for initialLayout means that we
      // don't care what previous layout the image was in. We want the image to be ready for 
      // presentation using the swap chain after rendering, which is why we use 
      // VK_IMAGE_LAYOUT_PRESENT_SRC_KHR as finalLayout.
      VkAttachmentDescription colorAttachment {};
      colorAttachment.format = swapChainImageFormat;
      colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
      colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

      // Reference to subpass
      // We intend to use the attachment to function as a color buffer
      VkAttachmentReference colorAttachmentRef {};
      colorAttachmentRef.attachment = 0;
      colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      // Subpass description
      VkSubpassDescription subpass {};
      subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.colorAttachmentCount = 1;
      subpass.pColorAttachments = &colorAttachmentRef;

      // Create the render pass object
      VkRenderPassCreateInfo renderPassInfo {};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
      renderPassInfo.attachmentCount = 1;
      renderPassInfo.pAttachments = &colorAttachment;
      renderPassInfo.subpassCount = 1;
      renderPassInfo.pSubpasses = &subpass;

      if(vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
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
      createInstance();         // initializes Vulkan library
      setupDebugMessenger();    // setup error handling
      createSurface();          // creates window surface
      pickPhysicalDevice();     // selects a suitable physical device
      createLogicalDevice();    // creates a logical device to interface with the physical device
      createSwapChain();        // create swap chain
      createImageViews();       // create the image views
      createRenderPass();       // create the render pass object
      createGraphicsPipeline(); // create the graphics pipeline
    }


    /// Resource management
    void cleanup() {
      // DO NOT CHANGE ORDER OF CLEANUP OF RESOURCES
      vkDestroyPipeline(logicalDevice, graphicsPipeline, nullptr);
      vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
      vkDestroyRenderPass(logicalDevice, renderPass, nullptr);

      for(auto imageView : swapChainImageViews) {
        vkDestroyImageView(logicalDevice, imageView, nullptr);
      }

      vkDestroySwapchainKHR(logicalDevice, swapChain, nullptr);
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
      // to keep the window open until it is closed or an error occurs
      while(!glfwWindowShouldClose(window)) { 
        glfwPollEvents();
      }
    }
    #pragma endregion
};


int main() {
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