def clean_tuple_str(tuple_str):

    tuple_str = tuple_str

    # only take the string before parenthesis
    tuple_str = tuple_str.strip().split('(')[0]

    tuple_str = tuple_str.strip()

    return tuple_str


def parse_tuple_output(output_str) -> dict:
    """Parse dependency gen result string into dict"""

    if 'output:' in output_str:
        start_index = output_str.index('output:')
        output_str = output_str[start_index+len('output:'):]
        output_str = output_str.strip()
        # print('refined: ', output_str)

    id2tup = {}
    for id_tup in output_str.strip().split('\n'):
        tup_id, tup = id_tup.split('|')

        tup_id = tup_id.strip()
        tup = tup.strip()

        tup = clean_tuple_str(tup)

        tup_id = int(tup_id)

        # tups = [int(d) for d in tup.split(',')]

        id2tup[tup_id] = tup

    return id2tup


def clean_dependency_id(dependency_id_str):

    dependency_ids = dependency_id_str

    # split with comma
    dependency_ids = dependency_ids.strip().split(',')

    # remove whitespace
    dependency_ids = [dep_id.strip() for dep_id in dependency_ids]

    # filter out string (e.g., '5, background' -> '5')
    # but keep '-'
    dependency_ids = [
        dep_id for dep_id in dependency_ids if dep_id.isnumeric() or dep_id == '-']

    # if includes 0 and others -> remove 0
    if len(dependency_ids) > 1:
        dependency_ids = [dep_id for dep_id in dependency_ids if dep_id != '0']

    dependency_ids = ','.join(dependency_ids)

    return dependency_ids


def parse_dependency_output(output_str) -> dict:
    """Parse dependency gen result string into dict"""

    if 'output:' in output_str:
        start_index = output_str.index('output:')
        output_str = output_str[start_index+len('output:'):]
        output_str = output_str.strip()
        # print('refined: ', output_str)

    id2dep = {}
    for id_dep in output_str.strip().split('\n'):
        question_id, dep = id_dep.split('|')

        question_id = question_id.strip()
        dep = dep.strip()

        dep = clean_dependency_id(dep)

        question_id = int(question_id)

        deps = [int(d) for d in dep.split(',')]

        id2dep[question_id] = deps

    return id2dep


def parse_question_output(output_str) -> dict:
    """Parse question gen result string into dict"""

    if 'output:' in output_str:
        start_index = output_str.index('output:')
        output_str = output_str[start_index+len('output:'):]
        output_str = output_str.strip()
        # print('refined: ', output_str)

    id2question = {}
    for id_question in output_str.strip().split('\n'):
        question_id, question = id_question.split('|')

        question_id = question_id.strip()
        question = question.strip()

        question_id = int(question_id)

        id2question[question_id] = question

    return id2question

    

