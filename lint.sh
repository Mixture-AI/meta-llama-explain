# Check if ufmt and ruff are installed.
if ! command -v ufmt &> /dev/null
then
    echo "ufmt could not be found"
    echo "Please install ufmt by running 'pip install ufmt'"
    exit
fi

if ! command -v ruff &> /dev/null
then
    echo "ruff could not be found"
    echo "Please install ruff by running 'pip install ruff'"
    exit
fi

# Check if arguments are provided.
if [ "$#" -eq 0 ]
then
    echo "Lint all files"
    ufmt format llama/
    ruff check llama/
else
    echo "Lint specific files"
    ufmt format "$@"
    ruff check "$@"
fi
