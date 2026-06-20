const books = [
  { title: 'Libro 1 · Gatos', lines: ['Los gatos son animales domésticos.', 'Los gatos tienen pelo.', 'Los gatos toman agua.', 'Los gatos pueden beber leche.', 'Los gatos son mascotas.'] },
  { title: 'Libro 2 · Perros', lines: ['Los perros son animales domésticos.', 'Los perros tienen pelo.', 'Los perros toman agua.', 'Los perros son mascotas.', 'Los perros ladran.'] },
  { title: 'Libro 3 · Animales', lines: ['Los animales necesitan agua.', 'Los animales necesitan alimento.', 'Los animales pueden ser mascotas.', 'Los animales tienen características físicas.'] }
];

const corpus = books.flatMap(book => book.lines).join(' ');
const cleanCorpus = corpus.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[¿?.,;]/g, '').replace(/\s+/g, ' ').trim();
const tokens = cleanCorpus.split(' ');
const vocabulary = [...new Set(tokens)].sort();
const tokenIds = Object.fromEntries(vocabulary.map((token, index) => [token, index + 1]));
const pipelineSteps = ['Libros', 'Corpus', 'Tokenización', 'Embeddings', 'Forward pass', 'Softmax', 'Loss', 'Backpropagation', 'Actualización', 'Modelo entrenado', 'Consulta', 'Respuesta'];
let currentStep = 0;
let userQuestion = '¿Qué es un gato?';
let autoTimer = null;

const semanticEpochs = [
  { epoch: 1, loss: 0.91, lr: 0.10, changed: 120, tokens: 18, insight: 'gato y perro: sin relación clara', sim: 0.12, point: 18 },
  { epoch: 2, loss: 0.70, lr: 0.08, changed: 220, tokens: 36, insight: 'comienzan a compartir contexto', sim: 0.34, point: 34 },
  { epoch: 3, loss: 0.42, lr: 0.06, changed: 310, tokens: 54, insight: 'se acercan semánticamente', sim: 0.58, point: 52 },
  { epoch: 4, loss: 0.20, lr: 0.04, changed: 430, tokens: 72, insight: 'muy similares', sim: 0.76, point: 68 },
  { epoch: 5, loss: 0.05, lr: 0.02, changed: 510, tokens: 90, insight: 'altamente relacionados', sim: 0.91, point: 78 }
];

function normalize(text) { return text.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[¿?]/g, ' ?').replace(/[.,;]/g, '').trim().split(/\s+/).filter(Boolean); }
function deterministicVector(seed, dims = 12) { return Array.from({ length: dims }, (_, i) => Number((Math.sin(seed * (i + 1)) * 0.72).toFixed(2))); }
function tokenChips(words) { return `<div class="token-flow">${words.map(w => `<span class="token-chip"><b>${w}</b><small>ID ${tokenIds[w] || 'UNK'}</small></span>`).join('<span class="data-arrow">→</span>')}</div>`; }
function vectorBlock(word, after = false) { const v = deterministicVector((tokenIds[word] || word.length) + (after ? 8 : 0), 10); return `<div class="vector-card"><b>${word}</b><div class="vector-bars">${v.map(n => `<i style="height:${Math.abs(n) * 42 + 8}px;background:${n >= 0 ? 'var(--cyan)' : 'var(--violet)'}"></i>`).join('')}</div><code>[${v.slice(0, 6).join(', ')}, ...]</code></div>`; }
function probabilityBars(items) { return `<div class="probabilities">${items.map(([name, value]) => `<div><label><span>${name}</span><b>${value}%</b></label><meter min="0" max="100" value="${value}"></meter></div>`).join('')}</div>`; }
function accordion(items) { return items.map((item, i) => `<details ${i < 2 ? 'open' : ''}><summary>${item.title}</summary><p>${item.text}</p></details>`).join(''); }
function mathCard({ name, goal, receives, returns, example, interpretation }) { return `<article class="math-card"><h3>${name}</h3><p><b>¿Para qué sirve?</b> ${goal}</p><p><b>¿Qué recibe?</b> ${receives}</p><p><b>¿Qué devuelve?</b> ${returns}</p><code>${example}</code><p><b>Interpretación:</b> ${interpretation}</p></article>`; }
function stateCard(mode, epoch) { const train = mode === 'training'; return `<div class="state-grid"><span>Modo<b>${train ? 'Entrenamiento' : 'Inferencia'}</b></span><span>Epoch<b>${train ? epoch.epoch : 'No existe'}</b></span><span>Loss<b>${train ? epoch.loss : 'No existe'}</b></span><span>Learning rate<b>${train ? epoch.lr : 'No existe'}</b></span><span>Pesos modificados<b>${train ? epoch.changed : '0 · congelados'}</b></span><span>Tokens procesados<b>${train ? epoch.tokens : normalize(userQuestion).length}</b></span><span>Gradiente<b>${train ? 'Calculado' : 'No existe'}</b></span><span>Backpropagation<b>${train ? 'Activa' : 'No existe'}</b></span></div>`; }
function semanticPanel(epochIndex = 0) { return `<div class="semantic-card"><p><b>Epoch ${semanticEpochs[epochIndex].epoch}</b>: ${semanticEpochs[epochIndex].insight}</p><div class="semantic-line"><span style="left:12%">gato</span><span class="dog" style="left:${semanticEpochs[epochIndex].point}%">perro</span></div><p>Similitud coseno gato↔perro: <b>${semanticEpochs[epochIndex].sim}</b></p></div>`; }
function embeddingsEvolution() { return `<div class="before-after"><section><h3>ANTES</h3>${vectorBlock('gatos')}${vectorBlock('perros')}</section><section><h3>DESPUÉS</h3>${vectorBlock('gatos', true)}${vectorBlock('perros', true)}</section></div>${semanticPanel(4)}`; }

const steps = [
  { phase:'Fase 1 · Construcción del conocimiento', sub:'Libros → corpus', goal:'Convertir textos humanos en materia prima observable.', mode:'Preparación', pipe:1, learn:0,
    brief:[['Qué ocurre','Los libros se reúnen como fuentes. Todavía no hay IA ni pesos aprendidos.'],['Qué entra','Frases humanas sobre gatos, perros, animales, agua y mascotas.'],['Qué sale','Un corpus textual que conserva patrones repetidos.'],['Qué observar','El conocimiento inicial está en el texto, no en el modelo.']],
    visual:()=>`<div class="book-grid">${books.map(b=>`<article><h3>${b.title}</h3>${b.lines.map(l=>`<p>${l}</p>`).join('')}</article>`).join('')}</div><div class="big-flow"><span>LIBROS</span><b>↓</b><span>CORPUS</span></div>`,
    math:()=>mathCard({ name:'Corpus = unión de documentos', goal:'Reunir ejemplos lingüísticos para detectar regularidades.', receives:'Documentos D₁, D₂, D₃.', returns:'Texto concatenado que será limpiado y tokenizado.', example:'Corpus = D₁ ∪ D₂ ∪ D₃', interpretation:'No calcula significado; solo organiza la evidencia textual.' }), state:()=>stateCard('prep', semanticEpochs[0]) },
  { phase:'Fase 2 · Representación', sub:'Tokenización y vocabulario', goal:'Traducir palabras a identificadores procesables.', mode:'Representación', pipe:3, learn:0,
    brief:[['Qué ocurre','El texto limpio se divide en tokens y cada token recibe un ID.'],['Por qué ocurre','La red opera con números, no con palabras directamente.'],['Qué cambia','La frase deja de ser texto libre y pasa a ser una secuencia discreta.'],['Qué no cambia','Los parámetros aún no aprenden.']],
    visual:()=>`<div class="text-transform"><p>${corpus}</p><b>↓ limpieza</b><p>${cleanCorpus}</p></div>${tokenChips(['los','gatos','toman','agua'])}<div class="vocab-cloud">${vocabulary.map(w=>`<span>${w}<small>${tokenIds[w]}</small></span>`).join('')}</div>`,
    math:()=>mathCard({ name:'Tokenización', goal:'Crear símbolos discretos que indexan vectores.', receives:'Texto normalizado: “los gatos toman agua”.', returns:'IDs: [' + ['los','gatos','toman','agua'].map(w=>tokenIds[w]).join(', ') + '].', example:'token_id("gatos") = ' + tokenIds.gatos, interpretation:'El ID no contiene significado; solo apunta a una fila de la matriz de embeddings.' }), state:()=>stateCard('prep', semanticEpochs[0]) },
  { phase:'Fase 2 · Representación', sub:'Embeddings', goal:'Convertir IDs en vectores ajustables.', mode:'Representación', pipe:4, learn:0,
    brief:[['Qué ocurre','Cada ID selecciona una fila de la matriz E.'],['Qué entra','IDs de tokens.'],['Qué sale','Vectores numéricos de muchas dimensiones.'],['Qué aprende','Aún nada: las coordenadas iniciales son arbitrarias.']],
    visual:()=>`<div class="big-flow"><span>ID gato</span><b>↓</b><span>fila E[gato]</span><b>↓</b><span>vector 100D</span></div><div class="vector-row">${['gatos','perros','mascotas'].map(w=>vectorBlock(w)).join('')}</div>`,
    math:()=>mathCard({ name:'Embedding', goal:'Representar tokens en un espacio continuo donde puedan compararse.', receives:'Un ID de token y la matriz E.', returns:'Un vector xᵢ = E[idᵢ].', example:'E[gato] = [0.12, 0.45, -0.22, ...]', interpretation:'Durante el entrenamiento, palabras con contextos similares moverán sus vectores hacia zonas cercanas.' }), state:()=>stateCard('prep', semanticEpochs[0]) },
  { phase:'Fase 3 · Aprendizaje', sub:'Forward pass y attention', goal:'Predecir el siguiente token antes de corregir el modelo.', mode:'Entrenamiento', pipe:5, learn:1,
    brief:[['Qué ocurre','Los embeddings fluyen por attention y Transformer hasta producir logits.'],['Por qué ocurre','Necesitamos una predicción para medir error.'],['Qué entra','“los gatos toman”.'],['Qué sale','Puntajes para posibles siguientes tokens.']],
    visual:()=>`<div class="process-lab"><span>Tokens</span><i></i><span>Embeddings</span><i></i><span class="active">Attention Q·K</span><i></i><span>Transformer</span><i></i><span>Logits</span></div><div class="attention-grid">${['los','gatos','toman'].map(r=>['los','gatos','toman'].map(c=>`<span class="${r==='toman'&&c==='gatos'?'hot':''}">${r}→${c}<b>${r===c?'.64':(c==='gatos'?'.31':'.08')}</b></span>`).join('')).join('')}</div>`,
    math:()=>mathCard({ name:'Attention(Q,K,V)', goal:'Determinar qué palabras son relevantes para cada token.', receives:'Q=lo que busco, K=lo que cada token ofrece, V=información que se mezclará.', returns:'Vectores contextuales ponderados.', example:'Para “toman”: gatos=0.80, los=0.12, toman=0.08', interpretation:'El modelo usa “gatos” para anticipar que el siguiente token plausible puede ser “agua”.' }), state:()=>stateCard('training', semanticEpochs[1]) },
  { phase:'Fase 3 · Aprendizaje', sub:'Softmax, loss y gradiente', goal:'Convertir puntajes en probabilidades y medir el error.', mode:'Entrenamiento', pipe:7, learn:2,
    brief:[['Qué ocurre','Softmax normaliza logits y Cross Entropy castiga la baja probabilidad del token correcto.'],['Qué entra','Logits del Transformer y objetivo “agua”.'],['Qué sale','Loss y gradientes.'],['Qué cambia','Aparece una señal que indica hacia dónde ajustar parámetros.']],
    visual:()=>`${probabilityBars([['agua',20],['leche',50],['pelo',15],['animal',15]])}<div class="error-flow"><span>Correcto: agua</span><b>error</b><span>Loss 0.91</span><b>gradiente ⇠</b><span>parámetros</span></div>`,
    math:()=>mathCard({ name:'Softmax + Cross Entropy', goal:'Comparar predicción con la respuesta correcta.', receives:'Logits z y etiqueta correcta y=agua.', returns:'Probabilidades y pérdida escalar.', example:'Loss = -log(P(agua)) = -log(0.40) = 0.91', interpretation:'Como la probabilidad de “agua” es baja, el modelo recibe una corrección fuerte.' }), state:()=>stateCard('training', semanticEpochs[2]) },
  { phase:'Fase 3 · Aprendizaje', sub:'Backpropagation y actualización', goal:'Modificar parámetros para reducir futuros errores.', mode:'Entrenamiento', pipe:9, learn:3,
    brief:[['Qué ocurre','La señal de error viaja hacia atrás por Softmax, Transformer, Attention y Embeddings.'],['Qué aprende','Aumenta la compatibilidad entre contextos “gatos toman” y “agua”.'],['Qué cambia','Embeddings y matrices Q,K,V,W se ajustan.'],['Qué no cambia','El corpus textual original no cambia.']],
    visual:()=>`<div class="backward"><span>Loss</span><i>←</i><span>Softmax</span><i>←</i><span>Transformer</span><i>←</i><span>Attention</span><i>←</i><span>Embeddings</span></div>${embeddingsEvolution()}`,
    math:()=>mathCard({ name:'Descenso por gradiente', goal:'Mover cada peso en dirección que reduzca la pérdida.', receives:'Peso actual θ, learning rate η y gradiente ∇L.', returns:'Nuevo peso actualizado.', example:'θ nuevo = θ - η∇L = 0.51 - 0.10×0.20 = 0.49', interpretation:'Aquí sí hay aprendizaje: los parámetros cambian por evidencia de error.' }), state:()=>stateCard('training', semanticEpochs[3]) },
  { phase:'Fase 4 · Entrenamiento completo', sub:'Epochs y espacio semántico', goal:'Observar cómo se estabiliza lo aprendido.', mode:'Entrenamiento', pipe:10, learn:4,
    brief:[['Qué ocurre','El dataset se recorre varias veces.'],['Qué cambia','La loss baja y los embeddings semánticamente relacionados se acercan.'],['Qué aprende','gato↔perro, gato↔mascota y perro↔animal comparten contexto.'],['Qué debe observar','Aprender no es memorizar una frase: es reorganizar parámetros.']],
    visual:()=>`<div class="loss-chart">${semanticEpochs.map(e=>`<span style="height:${e.loss*190+10}px"><b>E${e.epoch}</b><small>${e.loss}</small></span>`).join('')}</div>${embeddingsEvolution()}`,
    math:()=>mathCard({ name:'Similitud coseno', goal:'Medir cercanía semántica entre embeddings.', receives:'Dos vectores: e_gato y e_perro.', returns:'Un valor entre -1 y 1.', example:'cos(gato, perro): 0.12 → 0.91', interpretation:'La cercanía aumenta porque ambos aparecen como animales, mascotas, con pelo y tomando agua.' }), state:()=>stateCard('training', semanticEpochs[4]) },
  { phase:'Fase 5 · Inferencia', sub:'Consulta → respuesta', goal:'Usar parámetros congelados para generar texto.', mode:'Inferencia', pipe:11, learn:4,
    brief:[['Qué ocurre','La consulta se tokeniza y atraviesa el modelo entrenado.'],['Qué entra','Pregunta del usuario.'],['Qué sale','Una respuesta generada token por token.'],['Qué no existe','No hay loss, gradiente, backpropagation ni aprendizaje.']],
    visual:()=>`${tokenChips(normalize(userQuestion))}<div class="process-lab"><span>Prompt</span><i></i><span>Embeddings congelados</span><i></i><span>Attention</span><i></i><span>Softmax</span><i></i><span class="active">Respuesta</span></div>${probabilityBars([['animal',60],['mascota',20],['felino',15],['otro',5]])}<div class="answer">Un gato es un animal doméstico que puede ser mascota, tiene pelo y toma agua.</div>`,
    math:()=>mathCard({ name:'Generación autorregresiva', goal:'Elegir el siguiente token usando el contexto ya producido.', receives:'Prompt + tokens generados previamente + parámetros θ*.', returns:'Distribución del siguiente token.', example:'P(siguiente | “qué es un gato”) → animal: 0.60', interpretation:'La respuesta completa surge repitiendo forward pass; los pesos permanecen congelados.' }), state:()=>stateCard('inference', semanticEpochs[4]) },
  { phase:'Fase 6 · Comparación', sub:'Entrenamiento vs inferencia', goal:'Distinguir con precisión qué cambia y qué no cambia.', mode:'Comparación', pipe:12, learn:4,
    brief:[['Entrenamiento','Usa objetivos, calcula loss, gradientes y modifica pesos.'],['Inferencia','Usa pesos congelados para responder.'],['Idea clave','Predicción ocurre en ambos; aprendizaje solo ocurre en entrenamiento.'],['Resultado','El estudiante identifica entrada, salida, transformación y cambio de parámetros.']],
    visual:()=>`<table class="compare"><tr><th>Aspecto</th><th>Entrenamiento</th><th>Inferencia</th></tr>${[['Entrada','Corpus + objetivos','Prompt del usuario'],['Loss','Sí','No existe'],['Gradiente','Sí','No existe'],['Backpropagation','Sí','No existe'],['Aprendizaje','Sí, actualiza θ','No, θ congelado'],['Salida','Modelo más ajustado','Texto generado']].map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('')}</table>`,
    math:()=>mathCard({ name:'Dos objetivos distintos', goal:'Separar optimización de uso del modelo.', receives:'Entrenar: dataset. Inferir: prompt.', returns:'Entrenar: θ*. Inferir: respuesta.', example:'Entrenar = minimizar L(θ); Inferir = calcular P(y|x,θ*)', interpretation:'Si no se actualizan parámetros, el modelo no está aprendiendo en ese momento.' }), state:()=>stateCard('inference', semanticEpochs[4]) }
];

function render() {
  const step = steps[currentStep];
  phaseName.textContent = step.phase;
  subphaseName.textContent = step.sub;
  currentGoal.textContent = step.goal;
  stageMode.textContent = step.mode;
  stageTitle.textContent = step.sub;
  visualization.innerHTML = step.visual();
  learningBrief.innerHTML = accordion(step.brief.map(([title, text]) => ({ title, text })));
  mathPanel.innerHTML = step.math();
  modelState.innerHTML = step.state();
  aiLearning.innerHTML = semanticPanel(step.learn);
  pipeline.innerHTML = pipelineSteps.map((p, i) => `<li class="${i + 1 < step.pipe ? 'done' : i + 1 === step.pipe ? 'active' : ''}"><span>${i + 1}</span>${p}</li>`).join('');
  const progress = Math.round((currentStep / (steps.length - 1)) * 100);
  progressFill.style.width = `${progress}%`;
  progressText.textContent = `${progress}% · Paso ${currentStep + 1} de ${steps.length}`;
  [...phaseRail.children].forEach((button, i) => button.classList.toggle('active', i === currentStep));
}

function buildRail() { phaseRail.innerHTML = steps.map((step, i) => `<button data-step="${i}"><b>${step.phase.replace(' · ', '<br>')}</b><small>${step.sub}</small></button>`).join(''); }
function stopAuto() { clearInterval(autoTimer); autoTimer = null; autoBtn.textContent = 'Autoejecución'; }

buildRail();
phaseRail.addEventListener('click', event => { const button = event.target.closest('button[data-step]'); if (!button) return; currentStep = Number(button.dataset.step); render(); });
prevBtn.onclick = () => { currentStep = Math.max(0, currentStep - 1); render(); };
nextBtn.onclick = () => { currentStep = Math.min(steps.length - 1, currentStep + 1); render(); };
resetBtn.onclick = () => { stopAuto(); currentStep = 0; render(); };
autoBtn.onclick = () => { if (autoTimer) return stopAuto(); autoBtn.textContent = 'Pausar'; autoTimer = setInterval(() => { if (currentStep >= steps.length - 1) return stopAuto(); currentStep += 1; render(); }, Number(speedSelect.value)); };
speedSelect.onchange = () => { if (autoTimer) { stopAuto(); autoBtn.click(); } };
applyQuestion.onclick = () => { userQuestion = userQuestionInput.value || '¿Qué es un gato?'; render(); };

render();
