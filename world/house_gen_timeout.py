import signal

class TimeoutError(Exception):
	pass

def timeout(seconds):
	def decorator(func):
		def _handle_timeout(signum, frame):
			raise TimeoutError("Environment generation failed.")

		def wrapper(*args, **kwargs):
			# Set the signal handler for SIGALRM
			signal.signal(signal.SIGALRM, _handle_timeout)
			# Set the alarm to trigger after the specified timeout
			signal.alarm(seconds)
			try:
				# Call the original function
				result = func(*args, **kwargs)
			except TimeoutError as e:
				print("Timeout occurred:", e)
				result = None  # Set a default value or None as desired
			finally:
				# Reset the alarm
				signal.alarm(0)
				return result
		return wrapper
	return decorator
