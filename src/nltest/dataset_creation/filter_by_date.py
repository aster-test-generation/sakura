import json
import os
import re
import subprocess


class FilterByDate:
    def __init__(self):
        pass
    def filter(self, repo_path: str, date_str: str):
        """
            Returns all Java files added (i.e., first committed) after a given date in a Git repo.

            Args:
                date_str (str): Date in 'YYYY-MM-DD' format.
                repo_path (str): Path to the Git repository (default current directory).

            Returns:
                list: List of Java file paths added after the given date.
        """
        try:
            # Verify it's a Git repo
            subprocess.run(['git', '-C', repo_path, 'rev-parse', '--is-inside-work-tree'],
                           check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            raise Exception(f"{repo_path} is not a valid Git repository.")

            # Prepare the git log command
        git_command = [
            'git', '-C', repo_path, 'log',
            '--diff-filter=A',  # Filter only added files
            '--name-only',  # Show only file names
            '--pretty=format:',  # Suppress commit headers
            f'--since={date_str}',  # Filter by date
            '--', '*.java'  # Only Java files
        ]

        # Execute the git command
        result = subprocess.run(git_command, capture_output=True, text=True, check=True)

        # Split output into individual files and remove duplicates
        files = list({line.strip() for line in result.stdout.split('\n') if line.strip()})

        return files

    def classify_java_files(self, file_list):
        """
        Categorize files into application and test classes.
        """
        app_files = []
        test_files = []

        for f in file_list:
            # Heuristic classification
            if ('/test/' in f.lower()) or ('\\test\\' in f.lower()) or ('Test' in os.path.basename(f)):
                test_files.append(f)
            else:
                app_files.append(f)
        return app_files, test_files

    def count_tests_in_file(self, file_path, repo_path):
        """
        Count number of test methods in a Java file (based on @Test annotations).
        """
        abs_path = os.path.join(repo_path, file_path)
        if not os.path.exists(abs_path):
            return 0
        try:
            with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                return len(re.findall(r'@Test\b', content))
        except Exception:
            return 0

    def process_repos_in_directory(self, base_dir, date_str):
        """
        Iterate through subdirectories (each assumed to be a Git repo) and collect Java files added after date.
        Returns a dict ready for JSON serialization.
        """
        results = {}
        total_apps = 0
        total_tests = 0
        total_test_methods = 0

        for item in os.listdir(base_dir):
            repo_path = os.path.join(base_dir, item)
            if not os.path.isdir(repo_path):
                continue

            if not os.path.exists(os.path.join(repo_path, ".git")):
                print(f"Skipping non-git directory: {repo_path}")
                continue

            print(f"Processing repository: {repo_path}")
            java_files = self.filter(repo_path=repo_path, date_str=date_str)
            app, tests = self.classify_java_files(java_files)

            app_count = len(app)
            test_count = len(tests)

            # Count test methods
            repo_test_methods = sum(self.count_tests_in_file(t, repo_path) for t in tests)

            total_apps += app_count
            total_tests += test_count
            total_test_methods += repo_test_methods

            results[item] = {
                "application_classes": app,
                "test_classes": tests,
                "application_count": app_count,
                "test_count": test_count,
                "test_method_count": repo_test_methods
            }

        # Add overall summary
        results["summary"] = {
            "total_application_classes": total_apps,
            "total_test_classes": total_tests,
            "total_test_methods": total_test_methods
        }

        return results

if __name__ == '__main__':
    # filter = FilterByDate().filter(repo_path='/Users/rangeetpan/Documents/Research/Aster/nl2test/resources/datasets/commons-codec',
    #                                date_str='2025-01-31')
    base_dir = '/Users/rangeetpan/Documents/Research/Aster/nl2test/resources/datasets/'
    date_str = '2025-01-31'

    all_results = FilterByDate().process_repos_in_directory(base_dir, date_str)

    output_path = os.path.join(base_dir, "java_files_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nJSON report created at: {output_path}")
