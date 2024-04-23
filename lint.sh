# Check if ufmt and ruff are installed.
if ! command -v ufmt &> /dev/null
then
    echo "ufmt could not be found"
    echo "Please install ufmt by running 'pip install ufmt'"
    exit
fi

if ! command -v ruff &> /dev/null
then
    # https://docs.astral.sh/ruff/tutorial/
    echo "ruff could not be found"
    echo "Please install ruff by running 'pip install ruff'"
    exit
fi

# Check if arguments are provided.
if [ "$#" -eq 0 ]
then
    # Lint all files in the current directory, and fix any fixable errors.
    echo "Lint all files"
    ufmt format llama/
    ruff check llama/ --fix
else
    echo "Lint specific files"
    ufmt format "$@"
    ruff check "$@" --fix
fi
