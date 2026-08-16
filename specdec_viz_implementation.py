from fasthtml.common import *

colors = {'accepted':'#22c55e', 'rejected':'#ef4444', 'correction':'#3b82f6', 'committed':'#000000', 'guess':'#9ca3af'}

all_sentences = [
    # Scientific paper — n_accepted: 2, 5, 0  (3 iters)
    ({'prefix': ['Mean','reaction','times'], 'guesses': ['decreased','significantly','with','age','and'], 'n_accepted': 2, 'correction': 'across'},
     {'prefix': ['Mean','reaction','times','decreased','significantly','across'], 'guesses': ['all','four','experimental','conditions','after'], 'n_accepted': 5, 'correction': 'the'},
     {'prefix': ['Mean','reaction','times','decreased','significantly','across','all','four','experimental','conditions','after','the'], 'guesses': ['training','experiment','test','trial','period'], 'n_accepted': 0, 'correction': 'intervention'},
     ['Mean','reaction','times','decreased','significantly','across','all','four','experimental','conditions','after','the','intervention']),

    # Children's book — n_accepted: 4, 1, 3, 1  (3 iters)
    ({'prefix': ['Little','Bear','woke'], 'guesses': ['up','one','sunny','morning','and'], 'n_accepted': 4, 'correction': 'feeling'},
     {'prefix': ['Little','Bear','woke','up','one','sunny','morning','feeling'], 'guesses': ['very','hungry','so','he','went'], 'n_accepted': 1, 'correction': 'grumpy'},
     {'prefix': ['Little','Bear','woke','up','one','sunny','morning','feeling','very','grumpy'], 'guesses': ['because','he','had','not','slept'], 'n_accepted': 3, 'correction': 'no'},
     {'prefix': ['Little','Bear','woke','up','one','sunny','morning','feeling','very','grumpy','because','he','had','no'], 'guesses': ['porridge', 'to', 'eat', 'for', 'his'], 'n_accepted': 1, 'correction': '.'},
     ['Little','Bear','woke','up','one','sunny','morning','feeling','very','grumpy','because','he','had','no','porridge', '.']),

    # Legal clause — n_accepted: 0, 2, 1, 3, 2  (4 iters)
    ({'prefix': ['The','undersigned','parties'], 'guesses': ['agree','shall','must','will','may'], 'n_accepted': 0, 'correction': 'hereby'},
     {'prefix': ['The','undersigned','parties','hereby'], 'guesses': ['agree','to','the','following','terms'], 'n_accepted': 2, 'correction': 'indemnify'},
     {'prefix': ['The','undersigned','parties','hereby','agree','to','indemnify'], 'guesses': ['the','other','party','against','all'], 'n_accepted': 1, 'correction': 'Company'},
     {'prefix': ['The','undersigned','parties','hereby','agree','to','indemnify','the','Company'], 'guesses': ['against','any','and','every','claim'], 'n_accepted': 3, 'correction': 'all'},
     {'prefix': ['The','undersigned','parties','hereby','agree','to','indemnify','the','Company','against','any','and','all'], 'guesses': ['claims','arising','from','this','contract'], 'n_accepted': 2, 'correction': 'hereunder'},
     ['The','undersigned','parties','hereby','agree','to','indemnify','the','Company','against','any','and','all','claims','arising','hereunder']),

    # Literary fiction — n_accepted: 1, 3, 2, 4  (4 iters)
    ({'prefix': ['The','rain','had'], 'guesses': ['stopped','falling','all','night','long'], 'n_accepted': 1, 'correction': 'hours'},
     {'prefix': ['The','rain','had','stopped','hours'], 'guesses': ['ago','but','the','city','was'], 'n_accepted': 3, 'correction': 'streets'},
     {'prefix': ['The','rain','had','stopped','hours','ago','but','the','streets'], 'guesses': ['were','still','slick','with','water'], 'n_accepted': 2, 'correction': 'silver'},
     {'prefix': ['The','rain','had','stopped','hours','ago','but','the','streets','were','still','silver'], 'guesses': ['with','rain','under','the','moon'], 'n_accepted': 4, 'correction': 'streetlights'},
     ['The','rain','had','stopped','hours','ago','but','the','streets','were','still','silver','with','rain','under','the','streetlights']),
]


# the app only needs to be created once
app, rt = fast_app(pico=True)

def Tok(text, status, extra=''):
    return Span(text, style=f'background:{colors[status]};color:white;padding:2px 6px;margin:2px;border-radius:4px;white-space:nowrap;{extra}')

def tick_state(tick, iters):
    n_iters = len(iters)
    spec_iter = tick // 2           # which speculative iteration
    is_resolve = tick % 2 == 1      # propose or resolve?
    spec_done = spec_iter >= n_iters
    n_standard = (tick + 1) // 2    # standard reveals one token per odd tick
    return spec_iter, is_resolve, spec_done, n_standard

def wrap_row(name, row):
    return Div(Strong(name, style='flex:0 0 12em'),
               Div(row, style='flex:1;min-width:0;display:flex;flex-wrap:wrap'),
               style='display:flex;align-items:flex-start')

@rt
def step(tick: int = 0, sent: int=0):
    seed_length = 3         # number of tokens given as a seed at the start
    *iters, std_tokens = all_sentences[sent]
    spec_iter, is_resolve, spec_done, n_standard = tick_state(tick, iters)
    
    # determine phase for speculative decoding
    if is_resolve:
        phase = Span('M', Sub('p'), ' verifying (parallel)', style='white-space:nowrap') 
    else:
        phase = Span('M', Sub('q'), ' guessing', style='white-space:nowrap')
    if spec_done: phase = Span('finished', style='white-space:nowrap')

    # standard approach (without speculative decoding)
    # if n_standard == 0: std_row = Div(Em('(running…)', style='color:#9ca3af'))
    std_row = Div(*(Tok(t, 'committed') for t in std_tokens[:n_standard+seed_length]), style='min-width:0')

    if spec_done: 
        spec_row = Div(*(Tok(t, 'committed') for t in std_tokens), Span(phase, style='color:#9ca3af;margin-left:1em'), style='min-width:0')
    else:
        it = iters[spec_iter]
        prefix = [Tok(t, 'committed') for t in it['prefix']]
        if is_resolve:
            g = it['guesses']
            green = [Tok(t, 'accepted') for t in g[:it['n_accepted']]]      # accepted           
            red = [Tok(g[it['n_accepted']], 'rejected', 'text-decoration:line-through')] if it['n_accepted'] < len(g) else []
#             red = Tok(g[it['n_accepted']], 'rejected', 'text-decoration:line-through')      # first rejection
            blue = Tok(it['correction'], 'correction')                      # M_p's fix
            # g[n_accepted+1:] are discarded — not rendered
            spec_row = Div(*prefix, *green, red, blue, Span(phase, style='color:#9ca3af;margin-left:1em'), style='min-width:0')
        else:
            guesses = [Tok(t, 'guess') for t in it['guesses']]
            spec_row = Div(*prefix, *guesses, Span(phase, style='color:#9ca3af;margin-left:1em'), style='min-width:0')

    counter = Span(f'Step {tick//2 + 1} of {len(std_tokens) - seed_length}', style='margin:0 1em')
    back_btn = Button('◀ Back', hx_get=step.to(tick=tick-1, sent=sent), hx_target='#tracks', hx_swap='outerHTML', disabled=tick==0)
    next_btn = Button('Next ▶', hx_get=step.to(tick=tick+1, sent=sent), hx_target='#tracks', hx_swap='outerHTML', 
                disabled=tick+1>=2*(len(std_tokens)-seed_length))
    
    change_btn = Button('Change sentence', hx_get=step.to(tick=0, sent=(sent+1) % len(all_sentences)), hx_target='#tracks',    
                hx_swap='outerHTML')   

    return Div(wrap_row('Standard', std_row), Hr(), wrap_row('Speculative Decoding', spec_row), Hr(),
           Div(back_btn, counter, next_btn), Hr(), Div(change_btn), id='tracks', hx_swap='outerHTML')



@rt
def viz(): return Titled('Speculative Decoding Visualization', step(0))
