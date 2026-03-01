# Install the compiled extension
install(TARGETS your_extension DESTINATION your_package_name)

# Manually install the type marker and stubs if they exist
install(FILES 
    "src/your_package_name/py.typed"
    "src/your_package_name/your_extension.pyi"
    DESTINATION your_package_name
)