"""Parity-count obstruction for a*(a>>s)+{a,a>>s,C}.

This proves a narrow family is non-bijective, not a global hash lower bound.
"""
import json


def counts(width,shift):
    mask = (1<<width)-1
    values = range(1<<width)
    return {name:sum(((a*(a>>shift)+add(a))&mask)&1 for a in values)
            for name,add in [('a',lambda a:a),('b',lambda a:a>>shift),
                             ('even_constant',lambda a:0),('odd_constant',lambda a:1)]}


if __name__ == '__main__':
    cases = 0
    for width in range(2,13):
        for shift in range(1,width):
            result = counts(width,shift)
            expected = {'a':1<<(width-2),'b':1<<(width-2),
                        'even_constant':1<<(width-2),'odd_constant':3*(1<<(width-2))}
            assert result == expected,(width,shift,result)
            cases += 1
    print(json.dumps(dict(exhaustive_small_word_cases=cases,
                         full32_odd_outputs=[1<<30,3*(1<<30)],
                         permutation_required_odd_outputs=1<<31,
                         proof='For 1<=s<width, bits a[0] and a[s] are independent; '
                         'the four parity formulas have weight one or three, never two.'),indent=2))
