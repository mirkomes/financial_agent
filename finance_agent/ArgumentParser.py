import argparse as argparse_lib

class ArgumentParser:
    def __init__(self):
        self.parser = argparse_lib.ArgumentParser(
            description="Parse a single query or a batch file."
        )

        source_group = self.parser.add_mutually_exclusive_group(required=True)
        source_group.add_argument(
            "--prompt",
            type=str,
            help="Single user prompt.",
        )
        source_group.add_argument(
            "--batch_file_path",
            type=str,
            help="Path to a JSON file containing an array of prompts.",
        )

    def parse_args(self, argv=None):
        # Return the parsed command line arguments.
        return self.parser.parse_args(argv)
