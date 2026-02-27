cmake_minimum_required(VERSION 3.8)
project(llm_bt_builder)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 1. Generar el Servicio Personalizado (Esto ya crea el módulo Python para el srv)
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GenerateBT.srv"
)

# 2. Instalar el script ejecutable directamente
install(PROGRAMS
  llm_bt_builder/bt_agent_node.py
  llm_bt_builder/bt_client_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# 3. Instalar otros directorios
install(DIRECTORY config prompts launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()