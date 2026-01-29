'\nFancy Progress Bar\n'
from rich.progress import Progress,BarColumn,TextColumn,SpinnerColumn,TimeElapsedColumn,MofNCompleteColumn,TimeRemainingColumn
class ProgressBarFactory:
	'Produces a fancy progress bar'
	@classmethod
	def get_progress_bar(B):'Returns the progress bar';A=Progress(SpinnerColumn(),TextColumn('•'),TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),BarColumn(),TextColumn('• Completed/Total:'),MofNCompleteColumn(),TextColumn('• Elapsed:'),TimeElapsedColumn(),TextColumn('• Remaining:'),TimeRemainingColumn());return A