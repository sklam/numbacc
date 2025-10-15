set -xe
# Check if target directory already exists
if [ -d "$3" ]; then
    echo "Directory $3 already exists. Checking out branch $2..."
    cd "$3"
    git fetch origin
    git checkout "$2"
    git pull origin "$2"
else
    echo "Cloning repository to $3 with branch $2..."
    git clone $1 --branch="$2" --depth=1 "$3"
fi
