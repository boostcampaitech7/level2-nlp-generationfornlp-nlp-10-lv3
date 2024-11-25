# train_baseline.sh

# Check if config argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

# Get the config file from the argument
CONFIG=$1

# Check if the config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file '$CONFIG' does not exist."
    exit 1
fi

# Run the Python training script with the provided config file
python inference.py --config_path "$CONFIG"

# Exit with the same status as the Python script
exit $?