#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>  // automatically loads vulkan header along with GLFW's own definitions

#include <iostream>
#include <stdexcept>
#include <cstdlib>

#include <vector>
#include <cstring>

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

    /// Window creation 
    void initWindow() {
      glfwInit();  // init GLFW

      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // not create a window in a OpenGL context
      glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);    // disable window resizing

      window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }


    void initVulkan() {
      createInstance(); // initializes Vulkan library
      setupDebugMessenger();
    }


    void mainLoop() {
      while(!glfwWindowShouldClose(window)) {  // to keep the window open until it is closed or an error occurs
        glfwPollEvents();
      }
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


    /// Create an instance of the Vulkan library
    void createInstance() {
      // populate struct that holds info about our app
      VkApplicationInfo appInfo {}; 
      appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
      appInfo.pApplicationName   = "Vulkan Triangle No. 1";
      appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);  // constructs an API version number.
      appInfo.pEngineName        = "No Engine";
      appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
      appInfo.apiVersion         = VK_API_VERSION_1_0;  // #define VK_API_VERSION_1_0 VK_MAKE_VERSION(1, 0, 0)

      // Tells the Vulkan driver which global extensions we want to use
      VkInstanceCreateInfo createInfo{};
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
      if (enableValidationLayers && !checkValidationLayerSupport()) {
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