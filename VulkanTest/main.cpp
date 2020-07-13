#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h> // automatically loads vulkan header along with GLFW's own definitions

// necessary to make sure that glm functions use radians
#define GLM_FORCE_RADIANS
// for converting opengl [-1.0, 1.0] -> vulkan [0.0, 1.0] convention for depth buffering
#define GLM_FORCE_DEPTH_ZERO_TO_ONE 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

// add this line to avoid linker errors
#define STB_IMAGE_IMPLEMENTATION 
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdint> // Necessary for UINT32_MAX
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <optional> // from C++17
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>


// window parameters
const uint32_t  WIDTH                 = 800;
const uint32_t  HEIGHT                = 600;
const int       MAX_FRAMES_IN_FLIGHT  = 2;

const std::string MODEL_PATH          = "models/viking_room.obj";
const std::string TEXTURE_PATH        = "textures/viking_room.png";

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
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,
                                                                "vkCreateDebugUtilsMessengerEXT");
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
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, 
                                                                "vkDestroyDebugUtilsMessengerEXT");
  if(func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}
#pragma endregion


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


/// NOTES on swapchain
/// Simply checking if a swap chain is available is not sufficient, because it may not actually
/// be compatible with our window surface. Creating a swap chain also involves a lot more
/// settings than instance and device creation, so we need to query for some more details 
/// before we're able to proceed. There are basically 3 kinds of properties we need to check:
/// 1. Basic surface capabilities
///     - min / max number of images in swap chain 
///     - min / max width and height of images
/// 2. Surface formats (pixel format, color space)
/// 3. Available presentation modes
struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR        capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR>   presentModes;
};


#pragma region Vertex-data
struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  /// A vertex binding describes at which rate to load data from memory throughout the vertices.
  /// It specifies the number of bytes between data entries and whether to move to the next data
  /// entry after each vertex or after each instance.
  static VkVertexInputBindingDescription getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription {};
    bindingDescription.binding = 0; // the index of the binding in the array of bindings
    bindingDescription.stride = sizeof(Vertex); // the number of bytes from one entry to the next
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return bindingDescription;
  }

  /// An attribute description struct describes how to extract a vertex attribute from a chunk of
  /// vertex data originating from a binding description. We have two attributes, position and
  /// color, so we need two attribute description structs.
  static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions {};
    // Position attribute
    attributeDescriptions[0].binding = 0; // from which binding the per-vertex data comes
    attributeDescriptions[0].location = 0; // references the location directive in vertex shader
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT; // type of data for the attribute
                                              // implicitly defines the byte size of attribute data
    attributeDescriptions[0].offset = offsetof(Vertex, pos); // number of bytes since the start of
                                                             // the per-vertex data to read from

    // Color attribute
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    // TexCoord attribute
    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

    return attributeDescriptions;
  }

  bool operator==(const Vertex& other) const {
    return pos == other.pos && color == other.color && texCoord == other.texCoord;
  }
};

namespace std {
  template<> struct hash<Vertex> {
    size_t operator()(Vertex const& vertex) const {
      return ((hash<glm::vec3>()(vertex.pos) ^
              (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
              (hash<glm::vec2>()(vertex.texCoord) << 1);
    }
  };
}

//// interleaved vertex attributes here
//const std::vector<Vertex> vertices = {
//  // quad one
//  // CCW order of vertex data
//  // position               color               texCoord
//  {{-0.15f, -0.15f, 0.0f},  {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}}, // 0 - bottom left
//  {{0.15f, -0.15f, 0.0f},   {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}, // 1 - bottom right
//  {{0.15f, 0.15f, 0.0f},    {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}}, // 2 - top right
//  {{-0.15f, 0.15f, 0.0f},   {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}}, // 3 - top left
//
//  // quad two
//  {{-0.15f, -0.15f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}}, // 4
//  {{0.15f, -0.15f, -0.5f},  {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}, // 5
//  {{0.15f, 0.15f, -0.5f},   {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}}, // 6
//  {{-0.15f, 0.15f, -0.5f},  {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}}  // 7
//};


//// contents of the index buffer
//const std::vector<uint16_t> indices = {
//    0, 1, 2, 
//    2, 3, 0,
//
//    4, 5, 6,
//    6, 7, 4
//};


// Define the Uniform Buffer Object for the descriptor
struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 projection;
};
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
    GLFWwindow*                   window;
    VkInstance                    vkinstance;
    // The debug callback handle that needs to be explicitly created and destroyed.
    VkDebugUtilsMessengerEXT      debugMessenger;  
    // Implicitly destroyed, therefore need not do anything in cleanup.
    VkPhysicalDevice              physicalDevice  = VK_NULL_HANDLE;
    // A logical device instance. You can create multiple logical devices from the same physical 
    // device if you have varying requirements.
    VkDevice                      logicalDevice;                    

    // Should be destroyed before the logical device
    VkSwapchainKHR                swapChain;
    // To store the handles of the 'VKImage's that will be in the swap chain
    std::vector<VkImage>          swapChainImages;
    // Store the swapchain's surface format
    VkFormat                      swapChainImageFormat;
    // Store the swapchain's swap extent
    VkExtent2D                    swapChainExtent;
    // Store the image views.
    std::vector<VkImageView>      swapChainImageViews;

    // Device queues are implicitly cleaned up when the device is destroyed.
    // Store a handle to the graphics queue. 
    VkQueue                       graphicsQueue;
    // The presentation queue handle
    VkQueue                       presentQueue;   
    // VkSurfaceKHR is platform agnostic, but its creation isn't because it depends on window 
    // system details. It is destroyed before the application instance.
    VkSurfaceKHR                  surface;
    VkDescriptorSetLayout         descriptorSetLayout;
    VkPipelineLayout              pipelineLayout;
    VkRenderPass                  renderPass;
    VkPipeline                    graphicsPipeline;
    // Delete the framebuffers before the image views and render pass that they are based on, but
    // only after we've finished rendering
    std::vector<VkFramebuffer>    swapChainFramebuffers;
    VkCommandPool                 commandPool;
    std::vector<VkCommandBuffer>  commandBuffers;
    // Command buffers will be automatically freed when their command pool is destroyed, so we 
    // don't need an explicit cleanup.
    VkBuffer                      vertexBuffer;
    // Memory bound to the vertex buffer, must be freed after buffer is no longer being used
    VkDeviceMemory                vertexBufferMemory;
    VkBuffer                      indexBuffer;
    VkDeviceMemory                indexBufferMemory;
    std::vector<VkBuffer>         uniformBuffers;
    std::vector<VkDeviceMemory>   uniformBuffersMemory;
    // The descriptor pool should be destroyed when the swapchain is recreated because it depends
    // on the number of images
    VkDescriptorPool              descriptorPool;
    std::vector<VkDescriptorSet>  descriptorSets;
    std::vector<VkSemaphore>      imageAvailableSemaphores;
    std::vector<VkSemaphore>      renderFinishedSemaphores;
    std::vector<VkFence>          inFlightFences;
    /// If MAX_FRAMES_IN_FLIGHT is higher than the number of swap chain images or
    /// vkAcquireNextImageKHR returns images out-of-order then it's possible that we may start
    /// rendering to a swap chain image that is already in flight. To avoid this, we need to track
    /// for each swap chain image if a frame in flight is currently using it. This mapping will
    /// refer to frames in flight by their fences so we'll immediately have a synchronization 
    /// object to wait on before a new frame can use that image.
    std::vector<VkFence>          imagesInFlight;
    size_t                        currentFrame = 0;
    // For handling framebuffer resizes explicitly
    bool                          framebufferResized = false;

    VkImage                       textureImage;
    VkDeviceMemory                textureImageMemory;
    // Destroyed before the image itself
    VkImageView                   textureImageView;
    VkSampler                     textureSampler;

    VkImage                       depthImage;
    VkDeviceMemory                depthImageMemory;
    VkImageView                   depthImageView;

    std::vector<Vertex>           vertices;
    std::vector<uint32_t>         indices;
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

      // Verify that anisotropic filtering is supported
      VkPhysicalDeviceFeatures supportedFeatures;
      vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

      return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
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
      deviceFeatures.samplerAnisotropy = VK_TRUE;

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
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

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
        swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat,
                                                  VK_IMAGE_ASPECT_COLOR_BIT);

        //VkImageViewCreateInfo createInfo {};
        //createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        //createInfo.image = swapChainImages[i];
        //
        //// specify how the image data should be interpreted
        //createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        //createInfo.format = swapChainImageFormat;
        //
        //createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        //createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        //createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        //createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        //
        //// describes what the image's purpose is and which part of the image should be accessed
        //createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        //createInfo.subresourceRange.baseMipLevel = 0;
        //createInfo.subresourceRange.levelCount = 1;
        //createInfo.subresourceRange.baseArrayLayer = 0;
        //createInfo.subresourceRange.layerCount = 1;
        //
        ///// NOTES
        ///// If you were working on a stereographic 3D app, then you would create a swapchain with
        ///// multiple layers. You could then create multiple image views for each image representing
        ///// the views for the left and right eyes by accessing different layers.
        //
        //if(vkCreateImageView(logicalDevice, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
        //  throw std::runtime_error("failed to create image views!");
        //}
        //
        ///// add a similar loop to destroy the image views created now at the end of the program
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
      // VkPipelineShaderStageCreateInfo structures 
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
      auto bindingDescription = Vertex::getBindingDescription();
      auto attributeDescriptions = Vertex::getAttributeDescriptions();
      vertexInputInfo.vertexBindingDescriptionCount = 1;
      vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
      vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
      vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

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
      // outside the scissor rectangles will be discarded by the rasterizer. 
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
      rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
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
      colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT 
                                              | VK_COLOR_COMPONENT_B_BIT 
                                              | VK_COLOR_COMPONENT_A_BIT;
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

      // Enable depth testing
      VkPipelineDepthStencilStateCreateInfo depthStencil{};
      depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
      // Specifies if the depth of new fragments should be compared to the depth buffer to see if
      // they should be discarded
      depthStencil.depthTestEnable = VK_TRUE;
      // Specifies if the new depth of fragments that pass the depth test should actually be
      // written to the depth buffer
      depthStencil.depthWriteEnable = VK_TRUE;
      // Specifies the comparison that is performed to keep or discard fragments. We're sticking to
      // the convention of lower depth = closer
      depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
      // Optional depth bound test, which allows you to only keep fragments that fall within the 
      // specified depth range
      depthStencil.depthBoundsTestEnable = VK_FALSE;
      depthStencil.minDepthBounds = 0.0f; // Optional
      depthStencil.maxDepthBounds = 1.0f; // Optional
      depthStencil.stencilTestEnable = VK_FALSE;

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
      pipelineLayoutInfo.setLayoutCount = 1;
      pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
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
      pipelineInfo.pDepthStencilState = &depthStencil;
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
      /// A single color buffer attachment represented by one of the images from the swap chain.
      /// The 'format' of the color attachment should match the format of the swap chain images
      /// The 'loadOp' and 'storeOp' determine what to do with the data in the attachment before
      /// rendering and after rendering.
      /// The 'initialLayout' specifies which layout the image will have before the render pass 
      /// begins. The finalLayout specifies the layout to automatically transition to when the 
      /// render pass finishes. Using VK_IMAGE_LAYOUT_UNDEFINED for initialLayout means that we
      /// don't care what previous layout the image was in. We want the image to be ready for 
      /// presentation using the swap chain after rendering, which is why we use 
      /// VK_IMAGE_LAYOUT_PRESENT_SRC_KHR as finalLayout.
      VkAttachmentDescription colorAttachment {};
      colorAttachment.format = swapChainImageFormat;
      colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
      colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      // Reference to subpass, we intend to use the attachment to function as a color buffer
      VkAttachmentReference colorAttachmentRef {};
      colorAttachmentRef.attachment = 0;
      colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

      // Depth attachment
      VkAttachmentDescription depthAttachment{};
      depthAttachment.format = findDepthFormat();
      depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
      depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      // We don't care about storing depth data, it will not be used after drawing has finished
      depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      // We don't care about the previous depth contents
      depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      // Reference to subpass
      // Unlike color attachments, a subpass can only use a single depth (+stencil) attachment
      // The color attachment differs for every swap chain image, but the same depth image can be
      // used by all of them because only a single subpass is running at the same time due to our 
      // semaphores.
      VkAttachmentReference depthAttachmentRef{};
      depthAttachmentRef.attachment = 1;
      depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

      // Subpass description
      VkSubpassDescription subpass {};
      subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.colorAttachmentCount = 1;
      subpass.pColorAttachments = &colorAttachmentRef;
      subpass.pDepthStencilAttachment = &depthAttachmentRef;

      // Subpass dependency
      VkSubpassDependency dependency {};
      dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
      dependency.dstSubpass = 0;
      dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.srcAccessMask = 0;
      dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

      // Create the render pass object
      std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
      VkRenderPassCreateInfo renderPassInfo {};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
      renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
      renderPassInfo.pAttachments = attachments.data();
      renderPassInfo.subpassCount = 1;
      renderPassInfo.pSubpasses = &subpass;
      renderPassInfo.dependencyCount = 1;
      renderPassInfo.pDependencies = &dependency;

      if(vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
      }
    }
    #pragma endregion


    #pragma region Framebuffers
    /// NOTES on Framebuffers
    /// The attachments specified during render pass creation are bound by wrapping them into a
    /// 'VkFramebuffer' object. A framebuffer object references all of the VkImageView objects that
    /// represent the attachments. In our case it will be only a single one: the color attachment.
    /// However, the image that we have to use for the attachment depends on which image the
    /// swapchain returns when we retrieve one for presentation. That means that we have to create
    /// a framebuffer for all of the images in the swapchain and use the one that corresponds to
    /// the retrieved image at drawing time.
    /// We need to specify with which renderPass the framebuffer needs to be compatible. You can
    /// only use a framebuffer with the render passes that it is compatible with, which roughly
    /// means that they use the same number and type of attachments.
    void createFramebuffers() {
      // Resizing the container to hold all of the framebuffers
      swapChainFramebuffers.resize(swapChainImageViews.size());

      // Iterate through the image views and create framebuffers from them
      for(size_t i = 0; i < swapChainImageViews.size(); i++) {
        std::array<VkImageView, 2> attachments = {
          swapChainImageViews[i],
          depthImageView
        };

        VkFramebufferCreateInfo framebufferInfo {};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if(vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
          throw std::runtime_error("failed to create framebuffer!");
        }
      }
    }
    #pragma endregion


    #pragma region Command-buffers-and-command-pools
    /// NOTES on command buffers and command pools
    /// Commands in Vulkan, like drawing operations and memory transfers, are not executed directly
    /// using function calls. You have to record all of the operations you want to perform in
    /// command buffer objects. The advantage of this is that all of the hard work of setting up
    /// the drawing commands can be done in advance and in multiple threads. After that, you just
    /// have to tell Vulkan to execute the commands in the main loop.
    /// We have to create a command pool before we can create command buffers. Command pools manage
    /// the memory that is used to store the buffers and command buffers are allocated from them.
    ///
    /// Command buffers are executed by submitting them on one of the device queues, like the
    /// graphics and presentation queues we retrieved. Each command pool can only allocate command
    /// buffers that are submitted on a single type of queue. We're going to record commands for
    /// drawing, which is why we've chosen the graphics queue family.
    /// There are two possible flags for command pools :
    /// 1. VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: Hint that command buffers are rerecorded with new
    ///     commands very often(may change memory allocation behavior)
    /// 2. VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT : Allow command buffers to be rerecorded
    ///     individually, without this flag they all have to be reset together
    /// We will only record the command buffers at the beginning of the program and then execute
    /// them many times in the main loop, so we're not going to use either of these flags.
    void createCommandPool() {
      QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

      VkCommandPoolCreateInfo poolInfo {};
      poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
      poolInfo.flags = 0; // Optional

      if(vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
      }

      // We can now start allocating command buffers and recording drawing commands in them.
      // Because one of the drawing commands involves binding the right VkFramebuffer, we'll have
      // to record a command buffer for every image in the swap chain once again.
    }


    void createCommandBuffers() {
      commandBuffers.resize(swapChainFramebuffers.size());

      // Begin allocating command buffers 
      // The 'level' parameter specifies if the allocated command buffers are primary or secondary
      // command buffers.
      // 1. VK_COMMAND_BUFFER_LEVEL_PRIMARY: Can be submitted to a queue for execution, but cannot
      //    be called from other command buffers.
      // 2. VK_COMMAND_BUFFER_LEVEL_SECONDARY : Cannot be submitted directly, but can be called 
      //    from primary command buffers.
      VkCommandBufferAllocateInfo allocInfo {};
      allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      allocInfo.commandPool = commandPool;
      allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

      if(vkAllocateCommandBuffers(logicalDevice, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
      }
      
      // The range of depths in the depth buffer is 0.0 to 1.0 in Vulkan, where 1.0 lies at the far
      // view plane and 0.0 at the near view plane. The initial value at each point in the depth
      // buffer should be the furthest possible depth, which is 1.0.
      std::array<VkClearValue, 2> clearValues{};
      clearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
      clearValues[1].depthStencil = {1.0f, 0};

      // Begin recording command buffers
      for(size_t i = 0; i < commandBuffers.size(); i++) {
        // The 'flags' parameter specifies how we're going to use the command buffer. The following
        // values are available:
        // 1. VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: The command buffer will be rerecorded
        //    right after executing it once.
        // 2. VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT : This is a secondary command buffer
        //    that will be entirely within a single render pass.
        // 3. VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT : The command buffer can be resubmitted
        //    while it is also already pending execution.
        // None of these flags are applicable for us right now.
        // The 'pInheritanceInfo' parameter is only relevant for secondary command buffers. It
        // specifies which state to inherit from the calling primary command buffers.
        VkCommandBufferBeginInfo beginInfo {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0; // Optional
        beginInfo.pInheritanceInfo = nullptr; // Optional

        if(vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
          throw std::runtime_error("failed to begin recording command buffer!");
        }
        // If the command buffer was already recorded once, then calling vkBeginCommandBuffer will
        // implicitly reset it. It's not possible to append commands to a buffer at a later time.

        // Starting a render pass
        VkRenderPassBeginInfo renderPassInfo {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[i];
        // The next two parameters define the size of the render area. The render area defines
        // where shader loads and stores will take place. The pixels outside this region will have
        // undefined values. It should match the size of the attachments for best performance.
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;
        VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        // Begin render pass
        vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

          // Bind the graphics pipeline
          vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

          // Bind the vertex buffer during rendering operations
          VkBuffer vertexBuffers[] = {vertexBuffer};
          VkDeviceSize offsets[] = {0};
          vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

          // Bind the index buffers
          vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

          // Bind the right descriptor set for each swap chain image to the descriptors in the shader
          vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

          // Issue draw command
          // It has the following parameters, aside from the command buffer:
          // 1. vertexCount: We don't have a vertex buffer, but we have 3 vertices to draw.
          // 2. instanceCount : Used for instanced rendering, use 1 if you're not doing that.
          // 3. firstVertex : Used as an offset into the vertex buffer, defines the lowest value of
          //    gl_VertexIndex.
          // 4. firstInstance : Used as an offset for instanced rendering, defines the lowest value
          //    of gl_InstanceIndex.
          //vkCmdDraw(commandBuffers[i], static_cast<uint32_t>(vertices.size()), 1, 0, 0);
          // Use the indexed draw command now
          vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        // End render pass
        vkCmdEndRenderPass(commandBuffers[i]);

        // Finish recording the command buffer
        if(vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
          throw std::runtime_error("failed to record command buffer!");
        }
      }
    }
    #pragma endregion


    #pragma region Rendering-and-presentation
    /// NOTES
    /// This function performs the following operations:
    /// 1. Acquire an image from the swap chain
    /// 2. Execute the command buffer with that image as attachment in the framebuffer
    /// 3. Return the image to the swap chain for presentation
    /// Each of these events is executed asynchronously, but are set in a single function call.
    /// The function calls will return before the operations are actually finished and the order of
    /// execution is also undefined. That is unfortunate, because each of the operations depends on
    /// the previous one finishing. We are going to synchronize swap chain events using semaphores.
    /// We'll need one semaphore to signal that an image has been acquired and is ready for
    /// rendering, and another to signal that rendering has finished and presentation can happen.
    /// To perform CPU-GPU synchronization, Vulkan offers a second type of synchronization
    /// primitive called 'fences'. Fences are similar to semaphores in the sense that they can be 
    /// signaled and waited for, but this time we actually wait for them in our own code. In our
    /// code in drawFrame, initially we're waiting for a fence that has not been submitted. The
    /// problem here is that, by default, fences are created in the unsignaled state. That means
    /// that vkWaitForFences will wait forever if we haven't used the fence before. To solve that,
    /// we can change the fence creation to initialize it in the signaled state as if we had
    /// rendered an initial frame that finished.
    
    void drawFrame() {
      // Wait for the frame to be finished
      vkWaitForFences(logicalDevice, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

      // Acquire an image from the swap chain
      uint32_t imageIndex;
      VkResult result = vkAcquireNextImageKHR(logicalDevice, swapChain,
                            UINT64_MAX, // timeout in nanoseconds for an image to become available
                            imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

      if(result == VK_ERROR_OUT_OF_DATE_KHR) {
        // If the swap chain turns out to be out of date when attempting to acquire an image, then
        // it is no longer possible to present to it. We should immediately recreate the swap chain
        // and try again in the next drawFrame call.
        recreateSwapChain();
        return;
      }
      else if(result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
      }

      // Check if a previous frame is using this image (i.e. there is its fence to wait on)
      if(imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
        vkWaitForFences(logicalDevice, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
      }
      // Mark the image as now being in use by this frame
      imagesInFlight[imageIndex] = inFlightFences[currentFrame];

      updateUniformBuffer(imageIndex);

      // Achieve queue submission and synchronization through semaphores
      VkSubmitInfo submitInfo {};
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      // The first three parameters specify which semaphores to wait on before execution begins and
      // in which stage(s) of the pipeline to wait. We want to wait with writing colors to the
      // image until it's available, so we're specifying the stage of the graphics pipeline that
      // writes to the color attachment. That means that theoretically the implementation can
      // already start executing our vertex shader and such while the image is not yet available.
      VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
      VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
      submitInfo.waitSemaphoreCount = 1;
      submitInfo.pWaitSemaphores = waitSemaphores;
      submitInfo.pWaitDstStageMask = waitStages;
      // The next two parameters specify which command buffers to actually submit for execution. As
      // mentioned earlier, we should submit the command buffer that binds the swap chain image we
      // just acquired as color attachment.
      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

      VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
      submitInfo.signalSemaphoreCount = 1;
      submitInfo.pSignalSemaphores = signalSemaphores;

      vkResetFences(logicalDevice, 1, &inFlightFences[currentFrame]);
      if(vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
      }

      // Submit the result back to the swap chain
      VkPresentInfoKHR presentInfo {};
      VkSwapchainKHR swapChains[] = {swapChain};
      presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
      presentInfo.waitSemaphoreCount = 1;
      presentInfo.pWaitSemaphores = signalSemaphores;
      presentInfo.swapchainCount = 1;
      presentInfo.pSwapchains = swapChains;
      presentInfo.pImageIndices = &imageIndex;
      presentInfo.pResults = nullptr; // Optional

      result = vkQueuePresentKHR(presentQueue, &presentInfo);
      if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
      }
      else if(result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
      }

      vkQueueWaitIdle(presentQueue);

      currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    /// Create the semaphores and fences used in this application
    void createSyncObjects() {
      imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
      renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
      inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
      imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

      // semaphore info
      VkSemaphoreCreateInfo semaphoreInfo {};
      semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

      // fence info
      VkFenceCreateInfo fenceInfo {};
      fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

      for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if(vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
          vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
          vkCreateFence(logicalDevice, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
          throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
      }
    }
    #pragma endregion


    #pragma region Swapchain-recreation
    /// It is possible for the window surface to change such that the swap chain is no longer
    /// compatible with it. One of the reasons that could cause this to happen is the size of the
    /// window changing. We have to catch these events and recreate the swap chain.
    /// To handle window resizes properly, we also need to query the current size of framebuffer to
    /// make sure that the swap chain images have the (new) right size.
    /// Window minimization - a special case, which will result in a framebuffer size of 0.
    /// The disadvantage of this approach is that we need to stop all rendering before creating the
    /// new swap chain.
    void recreateSwapChain() {
      // Handle window minimization
      int width = 0, height = 0;
      glfwGetFramebufferSize(window, &width, &height);
      while(width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
      }

      // If resources are still in use
      vkDeviceWaitIdle(logicalDevice); 

      cleanupSwapchain();

      createSwapChain(); // recreate swapchain
      createImageViews(); // depends on swapchain images
      createRenderPass(); // depends on format of swapchain images
      createGraphicsPipeline(); // recreate viewport, scissor rectangles
      createDepthResources();
      createFramebuffers(); // depends on format of swapchain images
      createUniformBuffers(); // since we're destroying them every frame
      createDescriptorPool(); // since we're destroying them in cleanupSwapchain
      createDescriptorSets();
      createCommandBuffers(); // depends on format of swapchain images
    }


    /// Make sure that the old versions of these objects are cleaned up before recreating them
    void cleanupSwapchain() {
      vkDestroyImageView(logicalDevice, depthImageView, nullptr);
      vkDestroyImage(logicalDevice, depthImage, nullptr);
      vkFreeMemory(logicalDevice, depthImageMemory, nullptr);

      for(auto framebuffer : swapChainFramebuffers) {
        vkDestroyFramebuffer(logicalDevice, framebuffer, nullptr);
      }

      // clean up the existing command buffers with the vkFreeCommandBuffers function, reuse the
      // existing pool to allocate the new command buffers.
      vkFreeCommandBuffers(logicalDevice, commandPool, 
                            static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

      vkDestroyPipeline(logicalDevice, graphicsPipeline, nullptr);
      vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
      vkDestroyRenderPass(logicalDevice, renderPass, nullptr);

      for(auto imageView : swapChainImageViews) {
        vkDestroyImageView(logicalDevice, imageView, nullptr);
      }

      vkDestroySwapchainKHR(logicalDevice, swapChain, nullptr);

      for(size_t i = 0; i < swapChainImages.size(); i++) {
        vkDestroyBuffer(logicalDevice, uniformBuffers[i], nullptr);
        vkFreeMemory(logicalDevice, uniformBuffersMemory[i], nullptr);
      }

      vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
    }

    /// NOTES on recreation
    /// We need to figure out when swapchain recreation is necessary and call our new function.
    /// Luckily, Vulkan will usually just tell us that the swap chain is no longer adequate during
    /// presentation. The vkAcquireNextImageKHR and vkQueuePresentKHR functions can return the
    /// following special values to indicate this.
    /// 1. VK_ERROR_OUT_OF_DATE_KHR: The swap chain has become incompatible with the surface and 
    ///     can no longer be used for rendering. Usually happens after a window resize.
    /// 2. VK_SUBOPTIMAL_KHR : The swap chain can still be used to successfully present to the 
    ///     surface, but the surface properties are no longer matched exactly.

    /// GLFW callback for framebuffer resizes
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
      auto app = reinterpret_cast<TriangleApp*>(glfwGetWindowUserPointer(window));
      app->framebufferResized = true;
    }
    #pragma endregion


    #pragma region Vertex-and-Index-Buffers
    /// NOTES on Buffers
    /// Buffers in Vulkan are regions of memory used for storing arbitrary data that can be read by
    /// the graphics card, but buffers do not automatically allocate memory for themselves.
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                      VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
      VkBufferCreateInfo bufferInfo {};
      bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bufferInfo.size = size; // size of the buffer in bytes
      // 'usage' indicates for which purposes the data in the buffer is going to be used. It is 
      // possible to specify multiple purposes using a bitwise or.
      bufferInfo.usage = usage;
      // Buffers can also be owned by a specific queue family or be shared between multiple.
      // This buffer will only be used from the graphics queue.
      bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

      if(vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
      }

      // Query the memory requirements for the buffer
      VkMemoryRequirements memRequirements;
      vkGetBufferMemoryRequirements(logicalDevice, buffer, &memRequirements);

      // Memory allocation
      VkMemoryAllocateInfo allocInfo {};
      allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocInfo.allocationSize = memRequirements.size;
      allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

      if(vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
      }

      // Now associate this memory with the buffer
      // The fourth parameter (in the function) is the offset within the region of memory
      vkBindBufferMemory(logicalDevice, buffer, bufferMemory, 0);
    }


    /// Copy contents from one buffer to another
    /// Memory transfer operations are executed using command buffers, just like drawing commands.
    /// Therefore we must first allocate a temporary command buffer.
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
      VkCommandBuffer commandBuffer = beginSingleTimeCommands();

      VkBufferCopy copyRegion {};
      copyRegion.size = size;
      vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

      endSingleTimeCommands(commandBuffer);
    }

    
    void createVertexBuffer() {
      VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

      VkBuffer stagingBuffer;
      VkDeviceMemory stagingBufferMemory;

      createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                    stagingBuffer, stagingBufferMemory);

      // Copy the vertex data to the buffer, one by mapping the buffer memory into CPU accessible
      // memory with vkMapMemory
      void* data;
      vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t)bufferSize);
      vkUnmapMemory(logicalDevice, stagingBufferMemory);

      /// The driver may not immediately copy the data into the buffer memory, for example because
      /// of caching. It is also possible that writes to the buffer are not visible in the mapped
      /// memory yet. There are two ways to deal with that problem:
      /// 1. Use a memory heap that is host coherent - with VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
      /// 2. Call vkFlushMappedMemoryRanges after writing to the mapped memory, and call 
      ///    vkInvalidateMappedMemoryRanges before reading from the mapped memory
      /// Flushing memory ranges or using a coherent memory heap means that the driver will be aware
      /// of our writes to the buffer, but it doesn't mean that they are actually visible on the GPU
      /// yet. The transfer of data to the GPU is an operation that happens in the background and
      /// the specification simply tells us that it is guaranteed to be complete as of the next call
      /// to vkQueueSubmit.

      createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

      // The vertexBuffer is now allocated from a memory type that is device local, which generally
      // means that we're not able to use vkMapMemory. However, we can copy data from the
      // stagingBuffer to the vertexBuffer.
      copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

      // Cleanup the staging buffer
      vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
      vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }


    void createIndexBuffer() {
      VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

      VkBuffer stagingBuffer;
      VkDeviceMemory stagingBufferMemory;
      createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                    stagingBuffer, stagingBufferMemory);

      void* data;
      vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
      vkUnmapMemory(logicalDevice, stagingBufferMemory);

      createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

      copyBuffer(stagingBuffer, indexBuffer, bufferSize);

      vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
      vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }


    /// Function to find the right type of memory to use by combining the requirements of the
    /// buffer and our own application requirements.
    /// The VkPhysicalDeviceMemoryProperties structure has 2 arrays memoryTypes and memoryHeaps.
    /// Memory heaps are distinct memory resources like dedicated VRAM and swap space in RAM for
    /// when VRAM runs out. The different types of memory exist within these heaps.
    /// The memoryTypes array consists of VkMemoryType structs that specify the heap and
    /// properties of each type of memory.
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
      // Query info about the available types of memory
      VkPhysicalDeviceMemoryProperties memProperties;
      vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

      // The typeFilter parameter - specifies the bit field of memory types that are suitable.
      // So we can find the index of a suitable memory type by simply iterating over them and
      // checking if the corresponding bit is set to 1
      // We may have more than one desirable property, so we should check if the result of the
      // bitwise AND is not just non-zero, but equal to the desired properties bit field. If there
      // is a memory type suitable for the buffer that also has all of the properties we need, then
      // we return its index, otherwise we throw an exception.
      for(uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
          return i;
        }
      }

      throw std::runtime_error("failed to find suitable memory type!");
    }
    #pragma endregion


    #pragma region Uniform-Buffers
    /// NOTES on Uniform Buffers
    /// Resource Descriptors
    /// A descriptor is a way for shaders to freely access resources like buffers and images. We're
    /// going to set up a buffer that contains the transformation matrices and have the vertex 
    /// shader access them through a descriptor. Usage of descriptors consists of three parts:
    /// 1. A descriptor layout during pipeline creation - specifies the types of resources that are 
    ///     going to be accessed by the pipeline, just like a render pass specifies the types of 
    ///     attachments that will be accessed.
    /// 2. Allocate a descriptor set from a descriptor pool - specifies the actual buffer or image
    ///     resources that will be bound to the descriptors, just like a framebuffer specifies the 
    ///     actual image views to bind to render pass attachments.
    /// 3. Bind the descriptor set during rendering
    /// Uniform Buffer Objects (UBO) are a type of descriptor.
    /// We're going to copy new data to the uniform buffer every frame, so it doesn't really make 
    /// any sense to have a staging buffer. It would just add extra overhead in this case and
    /// likely degrade performance instead of improving it.
    /// We should have multiple buffers, because multiple frames may be in flight at the same time
    /// and we don't want to update the buffer in preparation of the next frame while a previous 
    /// one is still reading from it! We could either have a uniform buffer per frame or per swap 
    /// chain image. However, since we need to refer to the uniform buffer from the command buffer
    /// that we have per swap chain image, it makes the most sense to also have a uniform buffer
    /// per swap chain image.
    /// Using a UBO this way is not the most efficient way to pass frequently changing values to 
    /// the shader. A more efficient way to pass a small buffer of data to shaders are push 
    /// constants. 
    
    void createDescriptorSetLayout() {
      VkDescriptorSetLayoutBinding uboLayoutBinding {};
      uboLayoutBinding.binding = 0;
      uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      // It is possible for the shader variable to represent an array of uniform buffer objects, 
      // and descriptorCount specifies the number of values in the array.
      uboLayoutBinding.descriptorCount = 1;
      // Specify in which shader stages the descriptor is going to be referenced.
      uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
      // only relevant for image sampling related descriptors, therefore optional
      uboLayoutBinding.pImmutableSamplers = nullptr;

      // Combined image sampler
      VkDescriptorSetLayoutBinding samplerLayoutBinding {};
      samplerLayoutBinding.binding = 1;
      samplerLayoutBinding.descriptorCount = 1;
      samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      samplerLayoutBinding.pImmutableSamplers = nullptr;
      // we intend to use the combined image sampler descriptor in the fragment shader
      samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

      std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
      VkDescriptorSetLayoutCreateInfo layoutInfo {};
      layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
      layoutInfo.pBindings = bindings.data();

      if(vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
      }
    }


    void createUniformBuffers() {
      VkDeviceSize bufferSize = sizeof(UniformBufferObject);

      uniformBuffers.resize(swapChainImages.size());
      uniformBuffersMemory.resize(swapChainImages.size());

      for(size_t i = 0; i < swapChainImages.size(); i++) {
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                      uniformBuffers[i], uniformBuffersMemory[i]);
      }
    }


    /// Function to generate a new transformation every frame to make the geometry spin around.
    void updateUniformBuffer(uint32_t currentImage) {
      // calculate the time in seconds since rendering has started 
      static auto startTime = std::chrono::high_resolution_clock::now();

      auto currentTime = std::chrono::high_resolution_clock::now();
      float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

      // Now define the model, view and projection transformations in the uniform buffer object.
      // Here, model rotation is simply around Z-axis at a speed of 90 deg / second
      UniformBufferObject ubo {};
      ubo.model = glm::mat4(1.0f);
      //ubo.model = glm::rotate(glm::mat4(1.0f),              // identity matrix 
      //                        time * glm::radians(20.0f),   // rotation angle
      //                        glm::vec3(0.0f, 0.0f, 1.0f)); // rotation axis

      // View the geometry
      //ubo.view = glm::mat4(1.0f);
      ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), // eye position
                            glm::vec3(0.0f, 0.0f, 0.0f),  // center position
                            glm::vec3(0.0f, 0.0f, 1.0f)); // up axis

      /*const glm::mat4 clip(1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.5f, 0.0f,
        0.0f, 0.0f, 0.5f, 1.0f);*/

      // Perspective projection 
      ubo.projection = glm::perspective(glm::radians(45.0f), // field-of-view
                                  swapChainExtent.width / (float)swapChainExtent.height, // aspect ratio
                                  0.1f, // near plane
                                  10.0f); // far plane
      /// It is important to use the current swap chain extent to calculate the aspect ratio to take
      /// into account the new width and height of the window after a resize.

      /// GLM was originally designed for OpenGL, where the Y coordinate of the clip coordinates is
      /// inverted. The easiest way to compensate for that is to flip the sign on the scaling
      /// factor of the Y axis in the projection matrix. If you don't do this, then the image will 
      /// be rendered upside down.
      ubo.projection[1][1] *= -1;

      // Copy the data in the uniform buffer object to the current uniform buffer
      void* data;
      vkMapMemory(logicalDevice, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
      vkUnmapMemory(logicalDevice, uniformBuffersMemory[currentImage]);
    }


    /// Descriptor sets can't be created directly, they must be allocated from a pool like command
    /// buffers. The equivalent for descriptor sets is unsurprisingly called a 'descriptor pool'.
    /// This function will set it up.
    void createDescriptorPool() {
      // Describe which descriptor types our descriptor sets are going to contain and how many
      std::array<VkDescriptorPoolSize, 2> poolSizes {};
      poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
      poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      poolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

      VkDescriptorPoolCreateInfo poolInfo {};
      poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
      poolInfo.pPoolSizes = poolSizes.data();
      // Specify the maximum number of descriptor sets that may be allocated
      poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());
      // Optional flag similar to command pools that determines if individual descriptor sets can
      // be freed or not, we dont need it here.

      if(vkCreateDescriptorPool(logicalDevice, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
      }
    }


    void createDescriptorSets() {
      std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
      VkDescriptorSetAllocateInfo allocInfo {};
      allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      allocInfo.descriptorPool = descriptorPool;
      allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
      allocInfo.pSetLayouts = layouts.data();

      /// We will create one descriptor set for each swap chain image, all with the same layout.
      /// Unfortunately we do need all the copies of the layout because the next function expects an
      /// array matching the number of sets
      /// You don't need to explicitly clean up descriptor sets, because they will be automatically
      /// freed when the descriptor pool is destroyed.
      /// Unlike vertex and index buffers, descriptor sets are not unique to graphics pipelines. 
      /// Therefore we need to specify if we want to bind descriptor sets to the graphics or 
      /// compute pipeline.

      descriptorSets.resize(swapChainImages.size());
      // Allocate descriptor sets, each with one uniform buffer descriptor.
      if(vkAllocateDescriptorSets(logicalDevice, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
      }

      for(size_t i = 0; i < swapChainImages.size(); i++) {
        VkDescriptorBufferInfo bufferInfo {};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkDescriptorImageInfo imageInfo {};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;

        std::array<VkWriteDescriptorSet, 2> descriptorWrites {};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;
        descriptorWrites[0].pImageInfo = nullptr; // Optional
        descriptorWrites[0].pTexelBufferView = nullptr; // Optional
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), 
                                descriptorWrites.data(), 0, nullptr);
      }
    }
    #pragma endregion


    #pragma region Texture-mapping
    /// NOTES on Texture Mapping
    /// The geometry has been colored using per-vertex colors so far, which is a rather limited
    /// approach. So, we're going to implement texture mapping to make the geometry. Adding a 
    /// texture to our application will involve the following steps:
    /// 1. Create an image object backed by device memory
    /// 2. Fill it with pixels from an image file
    /// 3. Create an image sampler
    /// 4. Add a combined image sampler descriptor to sample colors from the texture
    /// Creating an image and filling it with data is similar to vertex buffer creation. We'll 
    /// start by creating a staging resource and filling it with pixel data and then we copy this
    /// to the final image object that we'll use for rendering. Images can have different layouts
    /// that affect how the pixels are organized in memory. Due to the way graphics hardware works,
    /// simply storing the pixels row by row may not lead to the best performance, for example.
    /// when performing any operation on images, you must make sure that they have the layout that
    /// is optimal for use in that operation. 
    /// One of the most common ways to transition the layout of an image is a pipeline barrier. 
    /// Pipeline barriers are primarily used for synchronizing access to resources, like making
    /// sure that an image was written to before it is read, but they can also be used to
    /// transition layouts.
    /// Image objects will make it easier and faster to retrieve colors by allowing us to use 2D
    /// coordinates, for one. Pixels within an image object are known as texels.
    /// It is possible to create 1D, 2D and 3D images. 1D images can be used to store an array of
    /// data or gradient, 2D images are mainly used for textures, and 3D images can be used to
    /// store voxel volumes, for example.


    /// Function to load an image and upload it into a Vulkan image object
    void createTextureImage() {
      // Load the image
      int texWidth, texHeight, texChannels;
      
      // STBI_rgb_alpha forces the image to be loaded with an alpha channel, even if it doesn't
      // have one, which is nice for consistency with other textures in the future.
      stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
      
      // The pointer that is returned by stbi_load is the first element in an array of pixel 
      // values. The pixels are laid out row by row with 4 bytes per pixel in the case of 
      // STBI_rgb_alpha for a total of texWidth * texHeight * 4 values.
      VkDeviceSize imageSize = texWidth * texHeight * 4;

      if(!pixels) {
        throw std::runtime_error("failed to load texture image!");
      }

      // Create a buffer in host visible memory, so that we can map it and it should be usable as a
      // transfer source so that we can copy it to an image later on
      VkBuffer stagingBuffer;
      VkDeviceMemory stagingBufferMemory;
      createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    stagingBuffer, stagingBufferMemory);

      // Directly copy the pixel values from the loaded image to the buffer
      void* data;
      vkMapMemory(logicalDevice, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
      vkUnmapMemory(logicalDevice, stagingBufferMemory);

      // Clean up the original pixel array now
      stbi_image_free(pixels);

      // Use the image creation helper function to create the image
      createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, 
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

      // Copy the staging buffer ot the texture image in 2 steps:
      // 1. Transition the texture image to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
      // 2. Execute the buffer to image copy operation
      transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, 
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), 
                        static_cast<uint32_t>(texHeight));

      // To be able to start sampling from the texture image in the shader, we need one last 
      // transition to prepare it for shader access
      transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, 
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

      vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
      vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }


    /// Helper function to create images
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
                      VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image,
                      VkDeviceMemory& imageMemory) {
      VkImageCreateInfo imageInfo {};
      imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      imageInfo.imageType = VK_IMAGE_TYPE_2D;
      imageInfo.extent.width = width;
      imageInfo.extent.height = height;
      imageInfo.extent.depth = 1;
      imageInfo.mipLevels = 1;
      imageInfo.arrayLayers = 1;
      // We should use the same format for the texels as the pixels, or the copy fails
      imageInfo.format = format;
      // The tiling mode cannot be changed at a later time. If you want to be able to directly
      // access texels in the memory of the image, then you must use VK_IMAGE_TILING_LINEAR
      imageInfo.tiling = tiling;
      // We're first going to transition the image to be a transfer destination and then copy texel
      // data to it from a buffer object, so we don't need the initialLayout property and can
      // safely use VK_IMAGE_LAYOUT_UNDEFINED
      imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      imageInfo.usage = usage;
      // multisampling
      imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
      imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      // There are some optional flags for images that are related to sparse images. Sparse images
      // are images where only certain regions are actually backed by memory. If you were using a
      // 3D texture for a voxel terrain, for example, then you could use this to avoid allocating
      // memory to store large volumes of "air" values. We won't be using it in this tutorial, so
      // leave it to its default value of 0.
      imageInfo.flags = 0; // Optional

      if(vkCreateImage(logicalDevice, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
      }

      // Allocate memory for image
      VkMemoryRequirements memRequirements;
      vkGetImageMemoryRequirements(logicalDevice, image, &memRequirements);

      VkMemoryAllocateInfo allocInfo {};
      allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocInfo.allocationSize = memRequirements.size;
      allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

      if(vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
      }

      vkBindImageMemory(logicalDevice, image, imageMemory, 0);
    }


    /// Helper functions for recording and executing command buffers
    VkCommandBuffer beginSingleTimeCommands() {
      VkCommandBufferAllocateInfo allocInfo {};
      allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocInfo.commandPool = commandPool;
      allocInfo.commandBufferCount = 1;

      VkCommandBuffer commandBuffer;
      vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer);

      VkCommandBufferBeginInfo beginInfo {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

      vkBeginCommandBuffer(commandBuffer, &beginInfo);

      return commandBuffer;
    }


    /// Helper functions for recording and executing command buffers
    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
      vkEndCommandBuffer(commandBuffer);

      VkSubmitInfo submitInfo {};
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers = &commandBuffer;

      vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
      vkQueueWaitIdle(graphicsQueue);

      vkFreeCommandBuffers(logicalDevice, commandPool, 1, &commandBuffer);
    }
    
    
    /// NOTES on Layout Transitions
    /// One of the most common ways to perform layout transitions is using an image memory barrier.
    /// A pipeline barrier like that is generally used to synchronize access to resources, like 
    /// ensuring that a write to a buffer completes before reading from it, but it can also be used
    /// to transition image layouts and transfer queue family ownership when 
    /// VK_SHARING_MODE_EXCLUSIVE is used. There is an equivalent buffer memory barrier to do this
    /// for buffers.


    /// Helper function to handle layout transitions
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, 
                                VkImageLayout newLayout) {
      // There are two transitions we need to handle:
      // 1. Undefined → transfer destination: transfer writes that don't need to wait on anything
      // 2. Transfer destination → shader reading: shader reads should wait on transfer writes, 
      //    specifically the shader reads in the fragment shader, because that's where we're going
      //    to use the texture
      VkPipelineStageFlags sourceStage;
      VkPipelineStageFlags destinationStage;
      VkCommandBuffer commandBuffer = beginSingleTimeCommands();

      VkImageMemoryBarrier barrier {};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      // The following two fields specify the layout transition. It is possible to use 
      // VK_IMAGE_LAYOUT_UNDEFINED as oldLayout if you don't care about the existing contents of
      // the image.
      barrier.oldLayout = oldLayout;
      barrier.newLayout = newLayout;
      // If you are using the barrier to transfer queue family ownership, then the next two fields
      // should be the indices of the queue families. They must be set to VK_QUEUE_FAMILY_IGNORED
      // if you don't want to do this
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      // The 'image' and 'subresourceRange' specify the image that is affected and the specific
      // part of the image.
      barrier.image = image;
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barrier.subresourceRange.baseMipLevel = 0;
      barrier.subresourceRange.levelCount = 1;
      barrier.subresourceRange.baseArrayLayer = 0;
      barrier.subresourceRange.layerCount = 1;
      // Barriers are primarily used for synchronization purposes, so you must specify which types
      // of operations that involve the resource must happen before the barrier, and which
      // operations that involve the resource must wait on the barrier. We need to do that despite
      // already using vkQueueWaitIdle to manually synchronize.
      //barrier.srcAccessMask = 0;
      //barrier.dstAccessMask = 0;
      if(oldLayout == VK_IMAGE_LAYOUT_UNDEFINED 
          && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      }
      else if(oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL 
              && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      }
      else if(oldLayout == VK_IMAGE_LAYOUT_UNDEFINED 
              && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT 
                                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
      }
      else {
        throw std::invalid_argument("unsupported layout transition!");
      }

      if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

        if (hasStencilComponent(format)) {
          barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
      } else {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      }

      // All types of pipeline barriers are submitted using the same function below
      // parameter 2 - specifies in which pipeline stage the operations occur that should happen 
      //  before the barrier
      // parameter 3 - specifies the pipeline stage in which operations will wait on the barrier
      // parameter 4 - cam be either 0 or VK_DEPENDENCY_BY_REGION_BIT which turns the barrier into
      //  a per-region condition. That means that the implementation is allowed to already begins
      //  reading from the parts of a resource that were written so far, for example      
      // parameters 5, 6 - arrays of pipeline barriers of type - memory barriers 
      // parameters 7, 8 - arrays of pipeline barriers of type - buffer memory barriers
      // parameters 9, 10 - arrays of pipeline barriers of type - image memory barriers 
      vkCmdPipelineBarrier(commandBuffer,
        sourceStage,
        destinationStage,
        0,
        0, nullptr, 
        0, nullptr, 
        1, &barrier);

      endSingleTimeCommands(commandBuffer);
    }


    /// Helper function to copy from buffer to image
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
      VkCommandBuffer commandBuffer = beginSingleTimeCommands();

      // Need to specify which part of the buffer is going to be copied to which part of the image
      VkBufferImageCopy region {};
      // byte offset in the buffer at which the pixel values start
      region.bufferOffset = 0;
      // Specify how the pixels are laid out in memory
      region.bufferRowLength = 0;
      region.bufferImageHeight = 0;
      // The imageSubresource, imageOffset and imageExtent fields indicate to which part of the
      // image we want to copy the pixels.
      region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.imageSubresource.mipLevel = 0;
      region.imageSubresource.baseArrayLayer = 0;
      region.imageSubresource.layerCount = 1;

      region.imageOffset = {0, 0, 0};
      region.imageExtent = {
          width,
          height,
          1
      };

      // Enqueue buffer to image copy operation
      vkCmdCopyBufferToImage(
        commandBuffer,
        buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
      );

      endSingleTimeCommands(commandBuffer);
    }


    /// NOTES on Samplers
    /// Textures are usually accessed through samplers, which will apply filtering and
    /// transformations to compute the final color that is retrieved. These filters are helpful to
    /// deal with problems like oversampling, which is preferred in conventional graphics
    /// applications. A sampler object automatically applies this filtering for you when reading a
    /// color from the texture. Undersampling is the opposite problem, where you have more texels
    /// than fragments. This will lead to artifacts when sampling high frequency patterns like a
    /// checkerboard texture at a sharp angle. The solution to this is 'anisotropic filtering',
    /// which can also be applied automatically by a sampler.
    

    /// Function to create a sampler object
    void createTextureSampler() {
      VkSamplerCreateInfo samplerInfo {};
      samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      // Specify how to interpolate texels that are magnified or minified, magnification concerns
      // the oversampling problem describes above, and minification concerns undersampling.
      samplerInfo.magFilter = VK_FILTER_LINEAR;
      samplerInfo.minFilter = VK_FILTER_LINEAR;
      // The addressing mode can be specified per axis using the 'addressMode' fields
      samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      // Specify if anisotropic filtering should be used.
      // Anisotropic filtering is actually an optional device feature. We need to update the
      // 'createLogicalDevice' function to request it and we should update 'isDeviceSuitable' to
      // check if it is available
      samplerInfo.anisotropyEnable = VK_TRUE;
      // Limits the amount of texel samples that can be used to calculate the final color. A lower
      // value results in better performance, but lower quality results. There is no graphics
      // hardware available today that will use more than 16 samples, because the difference is
      // negligible beyond that point.
      samplerInfo.maxAnisotropy = 16.0f;
      // Specifies which color is returned when sampling beyond the image with clamp to border
      // addressing mode. It is possible to return black, white or transparent in either float or
      // int formats. You cannot specify an arbitrary color.
      samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
      // Specifies which coordinate system you want to use to address texels in an image. If this
      // field is VK_TRUE, then you can simply use coordinates within the [0, texWidth) and 
      // [0, texHeight) range. If it is VK_FALSE, then the texels are addressed using the [0, 1)
      // range on all axes. Real-world applications almost always use normalized coordinates,
      // because then it's possible to use a variety of textures with the exact same coordinates.
      samplerInfo.unnormalizedCoordinates = VK_FALSE;
      // If a comparison function is enabled, then texels will first be compared to a value, and
      // the result of that comparison is used in filtering operations. This is mainly used for
      // percentage-closer filtering on shadow maps.
      samplerInfo.compareEnable = VK_FALSE;
      samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
      // Mipmapping
      samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
      samplerInfo.mipLodBias = 0.0f;
      samplerInfo.minLod = 0.0f;
      samplerInfo.maxLod = 0.0f;

      if(vkCreateSampler(logicalDevice, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
      }

      // The sampler does not reference a VkImage anywhere. The sampler is a distinct object that
      // provides an interface to extract colors from a texture. It can be applied to any image you
      // want, whether it is 1D, 2D or 3D. This is different from many older APIs, which combined
      // texture images and filtering into a single state.
    }


    /// NOTES on Combined Image Sampler
    /// A new type of descriptor which makes it possible for shaders to access an image resource
    /// through a sampler object.
    /// Modify the descriptor layout, descriptor pool and descriptor set to include such a combined
    /// image sampler descriptor. After that, add texture coordinates to 'Vertex' and modify the 
    /// fragment shader to read colors from the texture instead of just interpolating the vertex
    /// colors.


    /// Function to create an image view for our texture
    void createTextureImageView() {
      textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, 
                                          VK_IMAGE_ASPECT_COLOR_BIT);
    }


    /// Helper function to create image views
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
      VkImageViewCreateInfo viewInfo {};
      viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      viewInfo.image = image;
      viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      viewInfo.format = format;
      viewInfo.subresourceRange.aspectMask = aspectFlags;
      viewInfo.subresourceRange.baseMipLevel = 0;
      viewInfo.subresourceRange.levelCount = 1;
      viewInfo.subresourceRange.baseArrayLayer = 0;
      viewInfo.subresourceRange.layerCount = 1;

      VkImageView imageView;
      if(vkCreateImageView(logicalDevice, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
      }

      return imageView;
    }
    #pragma endregion


    #pragma region Depth-Buffering
    /// NOTES on depth buffering
    /// Approach 1 - Sort all of the draw calls by depth from back to front. This approach is
    /// commonly used for drawing transparent objects, because order-independent transparency is a
    /// difficult challenge to solve. However, the problem of ordering fragments by depth is much
    /// more commonly solved using a depth buffer. 
    /// Approach 2 - Use depth testing with a depth buffer, which is an additional attachment that
    /// stores the depth for every position, just like the color attachment stores the color of
    /// every position. Every time the rasterizer produces a fragment, the depth test will check if
    /// the new fragment is closer than the previous one. If it isn't, then the new fragment is
    /// discarded. A fragment that passes the depth test writes its own depth to the depth buffer.
    /// It is possible to manipulate this value from the fragment shader, just like you can 
    /// manipulate the color output.
    /// You will need to modify 'createRenderPass' also to include a depth attachment, and then
    /// modify the framebuffer creation to bind the depth image to the depth attachment. At this 
    /// point, the depth attachment is ready to be used, but depth testing still needs to be 
    /// enabled in the graphics pipeline.

    /// NOTES on Depth image and view
    /// A depth attachment is based on an image, just like the color attachment. The difference is
    /// that the swap chain will not automatically create depth images for us. We only need a
    /// single depth image, because only one draw operation is running at once. The depth image
    /// will again require the trifecta of resources : image, memory and image view.
    /// Creating a depth image is fairly straightforward. It should have the same resolution as the
    /// color attachment, defined by the swap chain extent, an image usage appropriate for a depth
    /// attachment, optimal tiling and device local memory.
    /// Unlike the texture image, we don't necessarily need a specific format, because we won't be
    /// directly accessing the texels from the program. It just needs to have reasonable accuracy,
    /// at least 24 bits is common in real-world applications.

    /// The depth buffer will be read from to perform depth tests to see if a fragment is visible,
    /// and will be written to when a new fragment is drawn. The reading happens in the 
    /// VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT stage and the writing in the 
    /// VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT. You should pick the earliest pipeline stage that
    /// matches the specified operations, so that it is ready for usage as depth attachment when it
    /// needs to be.

    /// Function to setup resources for the depth image
    void createDepthResources() {
      VkFormat depthFormat = findDepthFormat();

      createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, 
                  VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
      
      depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

      // Optional - transition the layout of the image to a depth attachment is handled in the
      // render pass
      transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, 
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }


    /// Helper function that takes a list of candidate formats in order from most desired to least
    /// desirable, and checks which is the first one that is supported. The support of a format 
    /// depends on the tiling mode and usage, so we must also include these as parameters.
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling,
                                  VkFormatFeatureFlags features) {
      for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

        if(tiling == VK_IMAGE_TILING_LINEAR 
            && (props.linearTilingFeatures & features) == features) {
          return format;
        }
        else if(tiling == VK_IMAGE_TILING_OPTIMAL 
                && (props.optimalTilingFeatures & features) == features) {
          return format;
        }
      }

      throw std::runtime_error("failed to find supported format!");
    }


    /// Helper function to select a format with a depth component that supports usage as depth
    /// attachment
    VkFormat findDepthFormat() {
      return findSupportedFormat(
          {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
          VK_IMAGE_TILING_OPTIMAL,
          VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
      );
    }


    /// Helper function that tells us if the chosen depth format contains a stencil component
    bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }
    #pragma endregion


    #pragma region Loading-models
    /// NOTES on OBJ models
    /// An OBJ file consists of positions, normals, texture coordinates and faces. Faces consist of
    /// an arbitrary amount of vertices, where each vertex refers to a position, normal and/or
    /// texture coordinate by index. This makes it possible to not just reuse entire vertices, but
    /// also individual attributes.
    /// The 'attrib' container holds all of the positions, normals and texture coordinates in its 
    /// attrib.vertices, attrib.normals and attrib.texcoords vectors. 
    /// The 'shapes' container contains all of the separate objects and their faces. Each face
    /// consists of an array of vertices, and each vertex contains the indices of the position,
    /// normal and texture coordinate attributes. OBJ models can also define a material and texture
    /// per face, but we will be ignoring those.
    /// The 'err' string contains errors and the warn string contains warnings that occurred while
    /// loading the file, like a missing material definition. Loading only really failed if the 
    /// 'LoadObj' function returns false. As mentioned above, faces in OBJ files can actually 
    /// contain an arbitrary number of vertices, whereas our application can only render triangles.
    /// Luckily the 'LoadObj' has an optional parameter to automatically triangulate such faces,
    /// which is enabled by default.
 

    /// Function that uses the tinyobjloader library to populate the vertices and indices containers
    /// with the vertex data from the mesh.
    void loadModel() {
      tinyobj::attrib_t attrib;
      std::vector<tinyobj::shape_t> shapes;
      std::vector<tinyobj::material_t> materials;
      std::string warn, err;

      if(!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
        throw std::runtime_error(warn + err);
      }

      // map to store unique vertices
      // using a user-defined type like our 'Vertex' struct as key in a hash table requires us to 
      // implement two functions: equality test and hash calculation. The former is easy to
      // implement by overriding the == operator in the 'Vertex' struct. A hash function for Vertex
      // is implemented by specifying a template specialization for std::hash<T>.
      std::unordered_map<Vertex, uint32_t> uniqueVertices{};

      // We're going to combine all of the faces in the file into a single model.
      // The triangulation feature has already made sure that there are three vertices per face, so
      // we can now directly iterate over the vertices and dump them straight into 'vertices'
      // For simplicity, we will assume that every vertex is unique for now.
      for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
          Vertex vertex{};

          // 'attrib.vertices' is an array of float values instead of something like glm::vec3, so
          // you need to multiply the index by 3
          vertex.pos = {
            attrib.vertices[3 * index.vertex_index + 0],
            attrib.vertices[3 * index.vertex_index + 1],
            attrib.vertices[3 * index.vertex_index + 2]
          };

          // The OBJ format assumes a coordinate system where a vertical coordinate of 0 means the
          // bottom of the image, however we've uploaded our image into Vulkan in a top to bottom
          // orientation where 0 means the top of the image. Solve this by flipping the vertical
          // component of the texture coordinates

          // there are two texture coordinate components per entry.
          vertex.texCoord = {
            attrib.texcoords[2 * index.texcoord_index + 0],
            1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
          };

          vertex.color = {1.0f, 1.0f, 1.0f};

          // The 'vertices' vector contains a lot of duplicated vertex data, because many vertices
          // are included in multiple triangles. We should keep only the unique vertices and use
          // the index buffer to reuse them whenever they come up.
          if(uniqueVertices.count(vertex) == 0) {
            uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
            vertices.push_back(vertex);
          }

          indices.push_back(uniqueVertices[vertex]);
        }
      }
    }
    #pragma endregion


    #pragma region Base-code
    /// Window creation 
    void initWindow() {
      glfwInit(); // init GLFW

      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // not create a window in a OpenGL context
      //glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);    // disable window resizing

      window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
      glfwSetWindowUserPointer(window, this);
      // Actually detect resizes
      glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }


    void initVulkan() {
      // ORDER OF FUNCTION CALLS SHOULD NOT BE CHANGED
      createInstance();             // initializes Vulkan library
      setupDebugMessenger();        // setup error handling
      createSurface();              // creates window surface
      pickPhysicalDevice();         // selects a suitable physical device
      createLogicalDevice();
      createSwapChain();
      createImageViews();
      createRenderPass();
      createDescriptorSetLayout();  // provide details about descriptor binding in shaders
      createGraphicsPipeline();
      createCommandPool();
      createDepthResources();
      createFramebuffers();         // create the framebuffers for our swapchain images
      createTextureImage();         // uses command buffers, hence called after createCommandPool
      createTextureImageView();
      createTextureSampler();
      loadModel();
      createVertexBuffer();
      createIndexBuffer();
      createUniformBuffers();
      createDescriptorPool();
      createDescriptorSets();
      createCommandBuffers();
      createSyncObjects();          // create the semaphores, fences for rendering and presentation
    }


    /// Resource management
    void cleanup() {
      // DO NOT CHANGE ORDER OF CLEANUP OF RESOURCES
      cleanupSwapchain();

      vkDestroySampler(logicalDevice, textureSampler, nullptr);
      vkDestroyImageView(logicalDevice, textureImageView, nullptr);
      vkDestroyImage(logicalDevice, textureImage, nullptr);
      vkFreeMemory(logicalDevice, textureImageMemory, nullptr);

      vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);

      vkDestroyBuffer(logicalDevice, indexBuffer, nullptr);
      vkFreeMemory(logicalDevice, indexBufferMemory, nullptr);

      vkDestroyBuffer(logicalDevice, vertexBuffer, nullptr);
      vkFreeMemory(logicalDevice, vertexBufferMemory, nullptr);

      for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(logicalDevice, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(logicalDevice, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(logicalDevice, inFlightFences[i], nullptr);
      }

      vkDestroyCommandPool(logicalDevice, commandPool, nullptr);
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
        drawFrame();
      }

      vkDeviceWaitIdle(logicalDevice);
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