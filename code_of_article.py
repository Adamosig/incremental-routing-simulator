import os, math, random, heapq
import networkx as nx, matplotlib.pyplot as plt, numpy as np

# --- КОНФИГУРАЦИЯ ---
random.seed(42)
np.random.seed(42)
os.makedirs('graphs', exist_ok=True)

SIZES =[100, 200, 300, 400, 500]
TRIALS = 15
ETA_VALUES = np.linspace(0, 0.3, 7)
# настройка генераторов
RGG_RADIUS, BA_M, SW_K, SW_P = 0.15, 3, 6, 0.3

# --- УТИЛИТЫ И СЧЕТЧИКИ ---
class OpCounter:
    def __init__(self): self.relax = self.push = self.pop = self.comp = 0
    def total_ops(self): return self.relax + self.push + self.pop + self.comp

def euclidean_heuristic(u, v, G):
    return math.dist(G.nodes[u]['pos'], G.nodes[v]['pos'])

def compute_initial_tree(G, source):
    try:
        lengths, paths = nx.single_source_dijkstra(G, source, weight='weight')
        d = {v: lengths.get(v, float('inf')) for v in G.nodes()}
        parent = {v: paths[v][-2] if len(paths[v]) > 1 else None for v in G.nodes()}
        return d, parent
    except Exception:
        return {v: 0 if v == source else float('inf') for v in G.nodes()}, {v: None for v in G.nodes()}

def find_descendants(parent, v):
    descendants, stack = set(), [v]
    while stack:
        node = stack.pop()
        for child, p in parent.items():
            if p == node and child not in descendants:
                descendants.add(child); stack.append(child)
    return descendants

# --- ГЕНЕРАТОРЫ ГРАФОВ ---
def assign_spatial_weights(G, mult_func=lambda: 1.0, min_w=0.0):
    for u, v in G.edges():
        w = math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']) * mult_func()
        G[u][v]['weight'] = max(min_w, w)

def generate_grid(n):
    side = math.ceil(math.sqrt(n))
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    if G.number_of_nodes() > n: G.remove_nodes_from(range(n, G.number_of_nodes()))
    nx.set_node_attributes(G, {node: (node%side, node//side) for node in G.nodes()}, 'pos')
    assign_spatial_weights(G, min_w=0.1)
    return G

def generate_rgg(n, radius=RGG_RADIUS):
    G = nx.convert_node_labels_to_integers(nx.random_geometric_graph(n, radius))
    assign_spatial_weights(G, lambda: random.uniform(1.0, 1.1))
    return G

def generate_scale_free(n, m=BA_M):
    G = nx.complete_graph(n) if n <= m else nx.barabasi_albert_graph(n, m)
    nx.set_node_attributes(G, {i: (random.random(), random.random()) for i in G.nodes()}, 'pos')
    assign_spatial_weights(G, lambda: random.uniform(5, 15))
    return G

def generate_small_world(n, k=SW_K, p=SW_P):
    G = nx.watts_strogatz_graph(n, k, p)
    nx.set_node_attributes(G, {i: (random.random(), random.random()) for i in G.nodes()}, 'pos')
    assign_spatial_weights(G, lambda: random.uniform(5, 15))
    return G

def generate_mesh(n, extra_edges_ratio=0.05):
    G = generate_grid(n)
    nodes = list(G.nodes())
    for _ in range(int(G.number_of_nodes() * extra_edges_ratio)):
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']))
    return G

generate_ami = lambda n: generate_rgg(n, RGG_RADIUS)

# --- БАЗОВЫЕ АЛГОРИТМЫ (Для сравнения) ---
def run_dijkstra_with_ops(G, source, c):
    d, parent = {n: float('inf') for n in G.nodes()}, {n: None for n in G.nodes()}
    d[source] = 0
    Q = [(0, source)]
    c.push += 1
    while Q:
        dist_u, u = heapq.heappop(Q)
        c.pop += 1; c.comp += 1
        if dist_u > d[u]: continue
        
        for v in G.neighbors(u):
            c.relax += 1; c.comp += 1
            new_dist = d[u] + G[u][v]['weight']
            if new_dist < d[v]:
                d[v], parent[v] = new_dist, u
                heapq.heappush(Q, (new_dist, v))
                c.push += 1
    return d, parent

def run_astar_with_ops(G, source, target, heuristic, c):
    closed_set, g_score = set(), {n: float('inf') for n in G.nodes()}
    parent, g_score[source] = {n: None for n in G.nodes()}, 0
    c.comp += 1
    open_set, open_set_nodes =[(heuristic(source, target), 0, source)], {source}
    c.push += 1
    
    while open_set:
        f_curr, g_curr, curr = heapq.heappop(open_set)
        c.pop += 1; c.comp += 1
        if g_curr > g_score[curr]: continue
        
        open_set_nodes.discard(curr)
        closed_set.add(curr)
        
        c.comp += 1
        if curr == target: return True # Нам важен лишь факт поиска и затраты
        
        for neighbor in G.neighbors(curr):
            c.relax += 1
            if neighbor in closed_set: continue
            
            tentative_g = g_score[curr] + G[curr][neighbor]['weight']
            c.comp += 1
            if tentative_g < g_score[neighbor]:
                parent[neighbor], g_score[neighbor] = curr, tentative_g
                c.comp += 1
                f_new = tentative_g + heuristic(neighbor, target)
                heapq.heappush(open_set, (f_new, tentative_g, neighbor))
                c.push += 1
                open_set_nodes.add(neighbor)
    raise nx.NetworkXNoPath()

# --- ПРЕДЛАГАЕМЫЙ ИНКРЕМЕНТАЛЬНЫЙ АЛГОРИТМ ---
def process_edge_decrease(G, d, parent, u, v, w_new, eta, c=None):
    A_dist, A_parent = set(), set()
    if c: c.comp += 1
    if d[u] + w_new < d[v] * (1 - eta):
        d[v], parent[v] = d[u] + w_new, u
        A_dist.add(v); A_parent.add(v)
        Q = [(d[v], v)]
        if c: c.push += 1
        
        while Q:
            dist_x, x = heapq.heappop(Q)
            if c: c.pop += 1; c.comp += 1
            if dist_x > d[x]: continue
            
            for y in G.neighbors(x):
                if c: c.relax += 1; c.comp += 1
                if d[x] + G[x][y]['weight'] < d[y] * (1 - eta):
                    old_d = d[y]
                    d[y] = d[x] + G[x][y]['weight']
                    if d[y] != old_d: A_dist.add(y)
                    if parent[y] != x: parent[y] = x; A_parent.add(y)
                    heapq.heappush(Q, (d[y], y))
                    if c: c.push += 1
    return A_dist, A_parent

def process_edge_increase(G, d, parent, u, v, w_new, eta, c=None):
    A_dist, A_parent = set(), set()
    if c: c.comp += 1
    if parent.get(v) != u: return A_dist, A_parent # Изменение не влияет на дерево
    
    # Фаза 1: Очистка поддерева
    descendants = find_descendants(parent, v)
    descendants.add(v)
    for z in descendants:
        if d[z] != float('inf'): A_dist.add(z)
        if parent.get(z) is not None: A_parent.add(z)
        d[z], parent[z] = float('inf'), None
    
    # Фаза 2: Поиск альтернатив через границу
    boundary_candidates =[]
    for z in descendants:
        for y in G.neighbors(z):
            if c: c.relax += 1
            if y not in descendants and d[y] != float('inf'):
                if c: c.comp += 1
                if d[y] + G[y][z]['weight'] < d[z]:
                    d[z], parent[z] = d[y] + G[y][z]['weight'], y
                    boundary_candidates.append(z)
                    A_dist.add(z); A_parent.add(z)
    
    # Фаза 3: Распространение волны
    if boundary_candidates:
        Q = [(d[z], z) for z in boundary_candidates]
        heapq.heapify(Q)
        if c: c.push += len(boundary_candidates)
        
        while Q:
            dist_x, x = heapq.heappop(Q)
            if c: c.pop += 1; c.comp += 1
            if dist_x > d[x]: continue
            
            for y in G.neighbors(x):
                if c: c.relax += 1; c.comp += 1
                if d[x] + G[x][y]['weight'] < d[y] * (1 - eta):
                    old_d = d[y]
                    d[y] = d[x] + G[x][y]['weight']
                    if d[y] != old_d: A_dist.add(y)
                    if parent[y] != x: parent[y] = x; A_parent.add(y)
                    heapq.heappush(Q, (d[y], y))
                    if c: c.push += 1
    return A_dist, A_parent

# --- ЭКСПЕРИМЕНТЫ ---
def experiment_complexity():
    results = {'sizes': [], 'full':[], 'astar': [], 'inc':[]}
    generators = {'Grid': generate_grid, 'RGG': generate_rgg, 
                  'ScaleFree': generate_scale_free, 'SmallWorld': generate_small_world}
    
    for n_nodes in SIZES:
        f_ops, a_ops, i_ops = [], [],[]
        for _ in range(TRIALS):
            g_type = random.choice(list(generators.keys()))
            G = generators[g_type](n_nodes)
            if G.number_of_nodes() < 2: continue
            
            nodes = list(G.nodes())
            src, target = nodes[0], nodes[-1]
            
            # 1. Dijkstra
            c_full = OpCounter()
            run_dijkstra_with_ops(G, src, c_full)
            f_ops.append(c_full.total_ops())
            
            # 2. A*
            c_astar = OpCounter()
            try:
                run_astar_with_ops(G, src, target, lambda u, v: euclidean_heuristic(u, v, G), c_astar)
                a_ops.append(c_astar.total_ops())
            except Exception:
                a_ops.append(n_nodes * 5)
                
            # 3. Инкрементальный
            edges = list(G.edges())
            if edges:
                u, v = random.choice(edges)
                old_w = G[u][v]['weight']
                new_w = old_w * random.uniform(0.5, 1.5)
                d_curr, parent_curr = compute_initial_tree(G, src)
                
                G[u][v]['weight'] = new_w
                c_inc = OpCounter()
                if new_w < old_w: process_edge_decrease(G, d_curr, parent_curr, u, v, new_w, 0.0, c_inc)
                else:             process_edge_increase(G, d_curr, parent_curr, u, v, new_w, 0.0, c_inc)
                i_ops.append(c_inc.total_ops())
                G[u][v]['weight'] = old_w
            else:
                i_ops.append(0)
                
        results['sizes'].append(n_nodes)
        results['full'].append(np.mean(f_ops))
        results['astar'].append(np.mean(a_ops))
        results['inc'].append(np.mean(i_ops))
    return results

def experiment_stability():
    results = {'eta': ETA_VALUES, 'churn': [], 'stretch':[]}
    n_nodes, n_events = 200, 30
    
    for eta in ETA_VALUES:
        ch_vals, st_vals = [],[]
        for _ in range(TRIALS):
            G = generate_grid(n_nodes)
            src = list(G.nodes())[0]
            d_opt, parent_opt = compute_initial_tree(G, src)
            d_curr, parent_curr = d_opt.copy(), parent_opt.copy()
            total_churn, total_stretch, st_count = 0, 0, 0
            
            for _ in range(n_events):
                edges = list(G.edges())
                if not edges: continue
                
                u, v = random.choice(edges)
                old_w = G[u][v]['weight']
                new_w = old_w * random.uniform(0.85, 1.15)
                G[u][v]['weight'] = new_w
                
                if new_w < old_w: _, A_parent = process_edge_decrease(G, d_curr, parent_curr, u, v, new_w, eta)
                else:             _, A_parent = process_edge_increase(G, d_curr, parent_curr, u, v, new_w, eta)
                
                total_churn += len(A_parent)
                
                for node in G.nodes():
                    if node != src and d_opt.get(node, float('inf')) != float('inf') and d_curr.get(node, float('inf')) != float('inf'):
                        d_opt_after, _ = compute_initial_tree(G, src)
                        stretch = max(1.0, d_curr[node] / d_opt_after[node])
                        total_stretch += stretch
                        st_count += 1
                G[u][v]['weight'] = old_w
                
            ch_vals.append(total_churn / n_events)
            if st_count > 0: st_vals.append(total_stretch / st_count)
            
        results['churn'].append(np.mean(ch_vals))
        results['stretch'].append(np.mean(st_vals) if st_vals else 1.0)
    return results

def experiment_mesh_ami():
    results = {'sizes': SIZES, 'mesh': [], 'ami':[]}
    for n_nodes in SIZES:
        affected = {'mesh': [], 'ami':[]}
        for topo_name, gen_func in [('mesh', generate_mesh), ('ami', generate_ami)]:
            for _ in range(TRIALS):
                try:
                    G = gen_func(n_nodes)
                    if G.number_of_nodes() < 2 or not G.edges(): continue
                    
                    src = list(G.nodes())[0]
                    u, v = random.choice(list(G.edges()))
                    old_w = G[u][v]['weight']
                    new_w = old_w * random.uniform(0.5, 1.5)
                    
                    d_curr, parent_curr = compute_initial_tree(G, src)
                    G[u][v]['weight'] = new_w
                    
                    if new_w < old_w: A_dist, _ = process_edge_decrease(G, d_curr, parent_curr, u, v, new_w, 0.0)
                    else:             A_dist, _ = process_edge_increase(G, d_curr, parent_curr, u, v, new_w, 0.0)
                    
                    affected[topo_name].append(len(A_dist))
                    G[u][v]['weight'] = old_w
                except Exception: pass
        results['mesh'].append(np.mean(affected['mesh']) if affected['mesh'] else 0)
        results['ami'].append(np.mean(affected['ami']) if affected['ami'] else 0)
    return results

def experiment_stress_test():
    ratios = np.linspace(0.01, 0.50, 10)
    results = {'ratios': ratios * 100, 'full': [], 'inc':[]}
    n_nodes, G_base = 400, generate_grid(400)
    src = list(G_base.nodes())[0]
    
    for r in ratios:
        f_ops, i_ops =[],[]
        for _ in range(10):
            G, edges = G_base.copy(), list(G_base.edges())
            k = int(len(edges) * r)
            if k == 0: continue
            removed = random.sample(edges, k)
            
            # Full Recompute
            G_after = G.copy()
            G_after.remove_edges_from(removed)
            c_full = OpCounter()
            if G_after.number_of_edges() > 0: run_dijkstra_with_ops(G_after, src, c_full)
            f_ops.append(c_full.total_ops() if G_after.number_of_edges() > 0 else n_nodes)
            
            # Incremental
            c_inc = OpCounter()
            d_curr, parent_curr = compute_initial_tree(G, src)
            for (u, v) in removed:
                G.remove_edge(u, v)
                process_edge_increase(G, d_curr, parent_curr, u, v, float('inf'), 0.0, c_inc)
            i_ops.append(c_inc.total_ops())
            
        results['full'].append(np.mean(f_ops))
        results['inc'].append(np.mean(i_ops))
    return results

# --- ПОСТРОЕНИЕ ГРАФИКОВ ---
def plot_results(res1, res2, res3, res4):
    def save_plot(name): plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(f'graphs/{name}.png', dpi=300); plt.close()
    
    # 1. Complexity
    plt.figure(figsize=(8, 5)); plt.title('Рис. 1. Сравнение вычислительной сложности')
    plt.plot(res1['sizes'], res1['full'], 'r-o', label='Dijkstra')
    plt.plot(res1['sizes'], res1['astar'], 'g--s', label='A*')
    plt.plot(res1['sizes'], res1['inc'], 'b-^', label='Proposed Incr')
    plt.xlabel('Размер сети |V|'); plt.ylabel('Операции'); plt.legend(); save_plot('exp1_complex')

    # 2. Churn & Stretch
    plt.figure(figsize=(8, 5)); plt.title('Рис. 2. Дрожание маршрутов от η')
    plt.plot(res2['eta'], res2['churn'], 'D-', color='purple')
    plt.xlabel('η'); plt.ylabel('Churn'); save_plot('exp2_churn')
    
    plt.figure(figsize=(8, 5)); plt.title('Рис. 3. Качество пути от η')
    plt.plot(res2['eta'], res2['stretch'], '^-', color='orange')
    plt.xlabel('η'); plt.ylabel('Stretch'); save_plot('exp2_stretch')

    # 3. Trade-off
    plt.figure(figsize=(8, 5)); plt.title('Рис. 4. Trade-off: Качество vs Стабильность')
    sc = plt.scatter(res2['stretch'], res2['churn'], c=res2['eta'], cmap='viridis', s=100, zorder=2)
    plt.plot(res2['stretch'], res2['churn'], 'k--', alpha=0.3, zorder=1)
    plt.colorbar(sc, label='Порог η'); plt.xlabel('Stretch'); plt.ylabel('Churn'); save_plot('exp2_tradeoff')

    # 4. Mesh vs AMI
    plt.figure(figsize=(8, 5)); plt.title('Рис. 5. Влияние топологии')
    plt.plot(res3['sizes'], res3['mesh'], 'b-o', label='Mesh')
    plt.plot(res3['sizes'], res3['ami'], 'r-s', label='AMI')
    plt.xlabel('|V|'); plt.ylabel('|A_dist|'); plt.legend(); save_plot('exp3_topo')

    # 5. Stress Test
    plt.figure(figsize=(8, 5)); plt.title('Рис. 6. Границы эффективности (Massive Failure)')
    plt.plot(res4['ratios'], res4['full'], 'r-o', label='Dijkstra')
    plt.plot(res4['ratios'], res4['inc'], 'b-s', label='Proposed Incr')
    plt.xlabel('% отказов'); plt.ylabel('Операции'); plt.legend(); save_plot('exp4_stress')

# --- ЗАПУСК ---
if __name__ == "__main__":
    print("Запуск экспериментов...")
    r1 = experiment_complexity(); print("ЭКСП 1 завершен")
    r2 = experiment_stability(); print("ЭКСП 2 завершен")
    r3 = experiment_mesh_ami(); print("ЭКСП 3 завершен")
    r4 = experiment_stress_test(); print("ЭКСП 4 завершен")
    plot_results(r1, r2, r3, r4)
    print("Графики сохранены в /graphs")