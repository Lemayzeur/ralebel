import base


while True:
	text = input('> ')
	result, error = base.run('base.opl', text)

	if error:
		print(error.as_string())
	else:
		print(result)