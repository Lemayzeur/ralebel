import base


while True:
	text = input('> ')
	result, error = base.run('base.bel', text)

	if error:
		print(error.as_string())
	else:
		if result: print(result) 