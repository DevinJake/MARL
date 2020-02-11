from symbolics_webqsp import Symbolics_WebQSP

seq74 = [
    {'A1': ['m.024v2j', 'people.person.parents', '?x']},
]
symbolic_exe = Symbolics_WebQSP(seq74)
answer = symbolic_exe.executor()
print("answer 74: ", answer)
