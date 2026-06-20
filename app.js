/*
  Simulador GPT educativo, escrito en JavaScript puro.
  La intención didáctica es mostrar los componentes reales de un LLM moderno
  con números pequeños y visualizaciones manejables para clase.
*/
const books = [
  { title: 'LIBRO 1 · LOS GATOS', lines: ['Los gatos son animales domésticos.', 'Los gatos tienen pelo.', 'Los gatos toman agua.', 'Los gatos pueden beber leche.', 'Los gatos son mascotas.'] },
  { title: 'LIBRO 2 · LOS PERROS', lines: ['Los perros son animales domésticos.', 'Los perros tienen pelo.', 'Los perros toman agua.', 'Los perros son mascotas.', 'Los perros ladran.'] },
  { title: 'LIBRO 3 · LOS ANIMALES', lines: ['Los animales necesitan agua.', 'Los animales necesitan alimento.', 'Los animales pueden ser mascotas.', 'Los animales tienen características físicas.'] }
];

const corpus = books.flatMap(book => book.lines).join(' ');
const cleanCorpus = corpus.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[¿?.,;]/g, '').replace(/\s+/g, ' ').trim();
const tokens = cleanCorpus.split(' ');
const vocabulary = [...new Set(tokens)].sort();
const tokenIds = Object.fromEntries(vocabulary.map((token, index) => [token, index + 1]));
const trainingExample = ['los', 'gatos', 'toman'];
let userQuestion = '¿Qué es un gato?';
let currentStep = 0;
let autoTimer = null;
let eventHistory = [];

const embeddings = Object.fromEntries(vocabulary.map((token, i) => [token, deterministicVector(i + 3, 100)]));
const learnedPositions = {
  gato: [31, 44], perro: [38, 47], animal: [52, 37], mascota: [45, 59], agua: [73, 68], leche: [78, 55]
};
const initialPositions = {
  gato: [14, 20], perro: [74, 18], animal: [49, 76], mascota: [22, 70], agua: [82, 42], leche: [35, 35]
};

function deterministicVector(seed, dims) {
  return Array.from({ length: dims }, (_, i) => Number((Math.sin(seed * (i + 1)) * 0.72).toFixed(2)));
}
function html(strings, ...values) { return strings.map((s, i) => s + (values[i] ?? '')).join(''); }
function tokenChips(words) { return `<div class="token-row">${words.map(w => `<span class="token">${w}<br><span class="id">ID ${tokenIds[w] || 'UNK'}</span></span>`).join('')}</div>`; }
function vectorPreview(word) { const v = embeddings[word] || deterministicVector(word.length, 100); return `<strong>${word}</strong><div class="embedding">${v.slice(0, 100).map(x => `<span title="${x}" style="opacity:${Math.abs(x) + .18}; background:${x >= 0 ? 'var(--accent)' : 'var(--accent-2)'}"></span>`).join('')}</div><small>[${v.slice(0, 8).join(', ')}, ... 100 valores]</small>`; }
function probabilityBars(items) { return items.map(([name, value]) => `<div class="bar"><label><span>${name}</span><b>${value}%</b></label><div class="meter"><span style="width:${value}%"></span></div></div>`).join(''); }
function attentionMatrix(words) { return `<table class="matrix"><tr><th></th>${words.map(w => `<th>${w}</th>`).join('')}</tr>${words.map((r, i) => `<tr><th>${r}</th>${words.map((_, j) => `<td class="${i === j || (r === 'gatos' && j === 2) ? 'hot' : ''}">${(0.12 + (i === j ? .52 : 0) + (j === 2 ? .18 : 0)).toFixed(2)}</td>`).join('')}</tr>`).join('')}</table>`; }
function space(points, title) { return `<h3>${title}</h3><div class="space">${Object.entries(points).map(([name, [x,y]]) => `<span class="point" style="left:${x}%;top:${y}%"><label>${name}</label></span>`).join('')}</div>`; }
function state(changing = false) { return `<div class="cards"><div class="card"><h3>Tokens</h3><p>${tokens.length}</p></div><div class="card"><h3>Vocabulario</h3><p>${vocabulary.length}</p></div><div class="card"><h3>Dimensión embedding</h3><p>100</p></div><div class="card"><h3>Estado de pesos</h3><p class="${changing ? 'changed' : 'frozen'}">${changing ? 'Cambian con gradiente' : 'Congelados / sin actualizar'}</p></div></div>`; }
function weights(changing = false) { const rows = [['W_emb[17]', .51, changing ? .49 : .51], ['W_q[34]', .23, changing ? .20 : .23], ['W_v[62]', -.18, changing ? -.12 : -.18]]; return `<div class="weights">${rows.map(([n,a,b]) => `<div class="weight"><b>${n}</b><p>${a} → <span class="${a!==b?'changed':'frozen'}">${b}</span></p></div>`).join('')}</div>`; }

const phases = [
  { m:'Módulo A', t:'Construcción del conocimiento', e:'Antes de existir IA solo hay textos humanos. Una fuente de datos es cada libro; el corpus será la colección que permitirá observar patrones estadísticos.', v:()=>`<div class="book-grid">${books.map(b=>`<div class="book"><h3>${b.title}</h3>${b.lines.map(l=>`<p>${l}</p>`).join('')}</div>`).join('')}</div><p class="formula">No existe inteligencia todavía. No existe aprendizaje todavía. Solamente existen libros escritos por humanos.</p>`, math:'Datos brutos = documentos escritos por humanos.', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo A', t:'Creación del corpus', e:'El corpus concatena documentos. Todavía no son ejemplos de entrenamiento: es materia prima textual que luego se transformará.', v:()=>`<div class="flow"><span class="node">Libro 1</span><span class="arrow">+</span><span class="node">Libro 2</span><span class="arrow">+</span><span class="node">Libro 3</span><span class="arrow">=</span><span class="node">Corpus</span></div><p>${corpus}</p>`, math:'Corpus = D₁ ∪ D₂ ∪ D₃', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo A', t:'Preprocesamiento', e:'Se normaliza el texto: minúsculas, limpieza de símbolos y espacios. Esto reduce variaciones superficiales y hace el vocabulario más consistente.', v:()=>`<h3>Antes</h3><p>${corpus}</p><h3>Después</h3><p>${cleanCorpus}</p>`, math:'lowercase(text) → remove(puntuación) → tokens limpios', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo A', t:'Tokenización', e:'Cada palabra se transforma en token y cada token recibe un ID numérico. El modelo no procesa letras como significado directo, sino índices que apuntan a vectores.', v:()=>tokenChips(['los','gatos','son','animales','domesticos']), math:'los=ID '+tokenIds.los+', gatos=ID '+tokenIds.gatos+', son=ID '+tokenIds.son, s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo A', t:'Vocabulario', e:'El vocabulario define el universo de símbolos conocidos. Los tokens fuera de este conjunto se tratarían como desconocidos o se dividirían con otro tokenizador.', v:()=>`<div class="token-row">${vocabulary.map(w=>`<span class="token">${w}<br><span class="id">${tokenIds[w]}</span></span>`).join('')}</div>`, math:'|V| = '+vocabulary.length, s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo A', t:'Inicialización aleatoria', e:'Los embeddings nacen como parámetros aleatorios. Todavía no codifican significado; solo son coordenadas ajustables en un espacio vectorial de 100 dimensiones.', v:()=>`${vectorPreview('gatos')}${vectorPreview('perros')}`, math:'E[token] ∈ ℝ¹⁰⁰, inicializado con valores pequeños', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo A', t:'Creación de ejemplos', e:'El entrenamiento causal crea pares entrada→objetivo para predecir el siguiente token. Esta tarea fuerza al modelo a aprender regularidades del lenguaje.', v:()=>`<div class="cards"><div class="card">Entrada: Los<br>Objetivo: gatos</div><div class="card">Entrada: Los gatos<br>Objetivo: son</div><div class="card">Entrada: Los gatos toman<br>Objetivo: agua</div></div>`, math:'P(xₜ | x₁,...,xₜ₋₁)', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo A', t:'Embeddings', e:'El ID del token selecciona una fila de la matriz de embeddings. Esa fila es una representación distribuida que podrá acercarse a palabras con contextos similares.', v:()=>trainingExample.map(vectorPreview).join('<hr>'), math:'xᵢ = E[idᵢ]', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo A', t:'Forward pass', e:'Los embeddings fluyen por capas Transformer para producir logits. En forward pass solo se calcula la predicción; aún no se corrigen pesos.', v:()=>`<div class="flow">${trainingExample.map(w=>`<span class="node">${w}</span><span class="arrow">→</span>`).join('')}<span class="node">Transformer</span><span class="arrow">→</span><span class="node">Logits</span></div>`, math:'logits = Transformer(E[los,gatos,toman])', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo A', t:'Self attention', e:'La atención permite que cada token pese la relevancia de otros tokens. “toman” atiende fuertemente a “gatos” y “los” para anticipar un objeto plausible.', v:()=>attentionMatrix(trainingExample), math:'Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo A', t:'Predicción', e:'El modelo asigna probabilidades. Aquí se equivoca: predice leche con más probabilidad que agua, aunque el objetivo del ejemplo es agua.', v:()=>probabilityBars([['agua',20],['leche',50],['animal',15],['pelo',15]])+'<p>Predicción: <b>leche</b> · Correcto: <b>agua</b></p>', math:'argmax(P)=leche, y*=agua', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo A', t:'Softmax', e:'Softmax convierte puntajes no normalizados en una distribución de probabilidad que suma 1.', v:()=>probabilityBars([['agua',20],['leche',50],['animal',15],['pelo',15]]), math:'softmax(zᵢ)=e^{zᵢ}/Σⱼe^{zⱼ}', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo A', t:'Función de pérdida', e:'La entropía cruzada penaliza asignar baja probabilidad al token correcto. Si P(agua)=0.40, la pérdida es -ln(0.40)=0.91.', v:()=>`<div class="card"><h3>Cross Entropy Loss</h3><p class="formula">Loss = -log(P(token correcto)) = -log(0.40) = 0.91</p></div>`, math:'L = -Σ yᵢ log(pᵢ)', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo A', t:'Gradientes', e:'El gradiente indica cómo cambiaría la pérdida si movemos cada parámetro. No todos contribuyen igual; se resaltan variables con señal relevante.', v:()=>`<div class="cards"><div class="card changed">Variable 17</div><div class="card changed">Variable 34</div><div class="card changed">Variable 62</div></div>`, math:'∂L/∂θ = dirección local de mayor aumento; se actualiza en sentido contrario', s:()=>state(true), p:()=>weights(false) },
  { m:'Módulo A', t:'Backpropagation', e:'La retropropagación aplica la regla de la cadena desde la pérdida hacia capas anteriores para calcular gradientes en embeddings y matrices Q,K,V.', v:()=>`<div class="flow"><span class="node">Loss</span><span class="arrow">←</span><span class="node">Softmax</span><span class="arrow">←</span><span class="node">Transformer</span><span class="arrow">←</span><span class="node">Embeddings</span></div>`, math:'∂L/∂θ₁ = (∂L/∂h)(∂h/∂θ₁)', s:()=>state(true), p:()=>weights(false) },
  { m:'Módulo A', t:'Actualización de pesos', e:'El optimizador modifica parámetros. Fórmula correcta: nuevo peso = peso actual − learning rate × gradiente. Aquí sí hay aprendizaje.', v:()=>weights(true), math:'θₙᵤₑᵥₒ = θ - η∇θL', s:()=>state(true), p:()=>weights(true) },
  { m:'Módulo A', t:'Epochs', e:'Una epoch recorre el dataset. Al repetir muchas veces, la pérdida baja si el modelo aprende patrones útiles sin memorizar de forma problemática.', v:()=>`<div class="loss-chart">${[[1,.91],[2,.70],[3,.42],[4,.20],[5,.05]].map(([e,l])=>`<div class="loss-bar" style="height:${l*170}px"><small>E${e}<br>${l}</small></div>`).join('')}</div>`, math:'Loss media por epoch: 0.91 → 0.70 → 0.42 → 0.20 → 0.05', s:()=>state(true), p:()=>weights(true) },
  { m:'Módulo A', t:'Embeddings aprendidos', e:'Gato y perro terminan cerca porque aparecen en contextos parecidos: animales domésticos, tienen pelo, toman agua y son mascotas. Nadie escribió una regla explícita.', v:()=>space(initialPositions,'Posición inicial')+space(learnedPositions,'Posición final aprendida'), math:'sim(gato, perro)=cos(e_gato,e_perro) aumenta por contextos compartidos', s:()=>state(false), p:()=>weights(true) },
  { m:'Módulo A', t:'Modelo entrenado', e:'Finalizado el entrenamiento, los parámetros quedan congelados para inferencia. La consulta no debe actualizar pesos ni calcular pérdida.', v:()=>`<div class="cards"><div class="card"><h3>Tokens</h3>${tokens.length}</div><div class="card"><h3>Parámetros didácticos</h3>${vocabulary.length * 100}</div><div class="card"><h3>Epochs</h3>5</div><div class="card"><h3>Loss final</h3>0.05</div></div>`, math:'θ* = parámetros entrenados y congelados', s:()=>state(false), p:()=>weights(true) },
  { m:'Módulo B', t:'Consulta: tokenización', e:'El usuario escribe una pregunta. Durante inferencia se tokeniza la entrada, pero no se crea etiqueta objetivo ni loss de entrenamiento.', v:()=>tokenChips(tokenizeQuestion()), math:'IDs consulta = ['+tokenizeQuestion().map(w=>tokenIds[w]||'UNK').join(', ')+']', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo B', t:'Consulta: embeddings', e:'Los tokens de la pregunta recuperan vectores ya aprendidos. Esos vectores son parámetros congelados.', v:()=>tokenizeQuestion().map(vectorPreview).join('<hr>'), math:'x_consulta = E_congelado[ids]', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo B', t:'Positional encoding', e:'El modelo necesita orden. El positional encoding se suma al embedding para distinguir “gato muerde perro” de “perro muerde gato”.', v:()=>`<table class="matrix"><tr><th>Token</th><th>Posición</th><th>Embedding + PE</th></tr>${tokenizeQuestion().map((w,i)=>`<tr><td>${w}</td><td>${i}</td><td>x${i}+PE${i}</td></tr>`).join('')}</table>`, math:'hᵢ⁰ = E[tokenᵢ] + PE(i)', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo B', t:'Self attention en consulta', e:'La palabra “gato” concentra peso porque contiene la entidad sobre la que se pregunta. Otras palabras estructuran la intención.', v:()=>attentionMatrix(tokenizeQuestion()), math:'pesos atención ≈ gato 0.80, qué 0.10, es 0.05, un 0.05', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo B', t:'Query, Key, Value', e:'Cada token se proyecta a Q, K y V. Q pregunta qué buscar, K anuncia qué contiene un token y V transporta la información que se mezcla.', v:()=>`<div class="qkv-grid">${tokenizeQuestion().map(w=>`<div class="card"><h3>${w}</h3><p>Q=Wq·x</p><p>K=Wk·x</p><p>V=Wv·x</p></div>`).join('')}</div>`, math:'Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo B', t:'Transformer contextual', e:'El Transformer produce representaciones contextuales. “banco” puede significar institución financiera o asiento según las palabras vecinas.', v:()=>`<div class="cards"><div class="card">banco + dinero → entidad financiera</div><div class="card">banco + parque → asiento</div></div>`, math:'hᵢᴸ = TransformerBlock(...TransformerBlock(hᵢ⁰))', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo B', t:'Softmax de salida', e:'La capa final estima el siguiente token más probable. No hay respuesta completa todavía: solo la próxima pieza textual.', v:()=>probabilityBars([['animal',60],['mascota',20],['felino',15],['otro',5]]), math:'P(siguiente token | pregunta) = softmax(W_out h)', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo B', t:'Generación autorregresiva', e:'Cada token generado se añade al contexto y vuelve a entrar al modelo. Así se produce una respuesta palabra por palabra.', v:()=>`<div class="flow">${['Un','gato','es','un','animal','doméstico'].map(w=>`<span class="node">${w}</span><span class="arrow">→</span>`).join('')}<span class="node">fin</span></div>`, math:'xₜ₊₁ ~ P(· | x₁...xₜ)', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo B', t:'Respuesta final', e:'Durante inferencia NO existe loss, gradiente, backpropagation, actualización de parámetros ni entrenamiento. Solo se usan los parámetros aprendidos previamente.', v:()=>`<div class="card"><h3>Respuesta generada</h3><p>Un gato es un animal doméstico que puede ser mascota, tiene pelo y toma agua.</p></div>`, math:'Solo forward pass repetido; θ permanece constante', s:()=>state(false), p:()=>weights(false) },
  { m:'Módulo C', t:'Comparación entrenamiento vs inferencia', e:'Entrenamiento ajusta parámetros usando datos, objetivos, loss y gradientes. Inferencia reutiliza parámetros congelados para generar predicciones.', v:()=>`<table class="compare"><tr><th>Aspecto</th><th>Entrenamiento</th><th>Inferencia</th></tr>${[['Dataset','Sí, corpus convertido en ejemplos','No, solo prompt del usuario'],['Loss','Sí','No'],['Gradiente','Sí','No'],['Backpropagation','Sí','No'],['Actualización de parámetros','Sí','No'],['Aprendizaje','Sí','No'],['Predicción','Sí, para medir error','Sí, para responder'],['Tiempo de ejecución','Costoso y prolongado','Rápido por consulta']].map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('')}</table>`, math:'Entrenar: minimizar L(θ). Inferir: calcular P(y|x,θ*)', s:()=>state(false), p:()=>weights(false) }
];

function tokenizeQuestion() { return userQuestion.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[¿?]/g, ' ?').replace(/[.,;]/g, '').trim().split(/\s+/).filter(Boolean); }
function render() {
  const phase = phases[currentStep];
  moduleBadge.textContent = phase.m; phaseTitle.textContent = phase.t; phaseCounter.textContent = `Paso ${currentStep + 1} de ${phases.length}`;
  explanation.innerHTML = `<p>${phase.e}</p>`; visualization.innerHTML = phase.v(); mathPanel.innerHTML = `<div class="formula">${phase.math}</div>`; statePanel.innerHTML = phase.s(); paramsPanel.innerHTML = phase.p();
  const progress = Math.round((currentStep / (phases.length - 1)) * 100); progressBar.value = progress; progressText.textContent = `${progress}%`;
  [...timeline.children].forEach((btn, i) => btn.classList.toggle('active', i === currentStep));
  addEvent(`${phase.m} · ${phase.t}`);
}
function addEvent(text) { eventHistory.unshift(`${new Date().toLocaleTimeString()} — ${text}`); eventHistory = eventHistory.slice(0, 14); eventLog.innerHTML = eventHistory.map(e => `<li>${e}</li>`).join(''); }
function buildTimeline() { timeline.innerHTML = phases.map((p, i) => `<button data-step="${i}">${i + 1}. ${p.t}</button>`).join(''); timeline.addEventListener('click', e => { if (e.target.dataset.step) { currentStep = Number(e.target.dataset.step); render(); } }); }
function stopAuto() { clearInterval(autoTimer); autoTimer = null; autoBtn.textContent = 'Autoejecución'; }

prevBtn.onclick = () => { currentStep = Math.max(0, currentStep - 1); render(); };
nextBtn.onclick = () => { currentStep = Math.min(phases.length - 1, currentStep + 1); render(); };
resetBtn.onclick = () => { stopAuto(); currentStep = 0; eventHistory = []; render(); };
autoBtn.onclick = () => { if (autoTimer) return stopAuto(); autoBtn.textContent = 'Pausar autoejecución'; autoTimer = setInterval(() => { if (currentStep >= phases.length - 1) return stopAuto(); currentStep += 1; render(); }, Number(speedSelect.value)); };
speedSelect.onchange = () => { if (autoTimer) { stopAuto(); autoBtn.click(); } };
applyQuestion.onclick = () => { userQuestion = document.getElementById('userQuestion').value || '¿Qué es un gato?'; render(); };
const userQuestionInput = document.getElementById('userQuestion');

buildTimeline();
render();
