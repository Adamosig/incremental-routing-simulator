import os
import math
import random
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# --- КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТОВ ---
random.seed(42)
np.random.seed(42)
os.makedirs('graphs', exist_ok=True)

SIZES =[100, 200, 300, 400, 500]
TRIALS = 15
ETA_VALUES = np.linspace(0, 0.3, 7)
RGG_RADIUS, BA_M, SW_K, SW_P = 0.15, 3, 6, 0.3

# --- СТРУКТУРЫ ДАННЫХ И СЧЕТЧИКИ ---
class OpCounter:
    """Унифицированный счетчик базовых вычислительных операций."""
    def __init__(self):
        self.relax = 0  # Количество проверок/релаксаций ребер
        self.pop = 0    # Количество извлечений из очереди / посещений узлов

    def total_ops(self):
        return self.relax + self.pop

def euclidean_heuristic(u, v, G):
    """Евклидова эвристика для алгоритма A*."""
    return math.dist(G.nodes[u]['pos'], G.nodes[v]['pos'])

def update_parent(v, new_p, parent, children):
    """O(1) обновление родителя с синхронизацией дерева потомков."""
    old_p = parent.get(v)
    if old_p == new_p: 
        return
    
    if old_p is not None and old_p in children:
        children[old_p].discard(v)
        
    parent[v] = new_p
    if new_p is not None:
        if new_p not in children:
            children[new_p] = set()
        children[new_p].add(v)

def compute_initial_tree(G, source):
    """Эталонный пересчет дерева кратчайших путей."""
    d = {v: float('inf') for v in G.nodes()}
    parent = {v: None for v in G.nodes()}
    children = {v: set() for v in G.nodes()}
    
    d[source] = 0
    Q = [(0, source)]
    
    while Q:
        dist_u, u = heapq.heappop(Q)
        if dist_u > d[u]: continue
        
        for v in G.neighbors(u):
            w = G[u][v]['weight']
            if d[u] + w < d[v]:
                d[v] = d[u] + w
                update_parent(v, u, parent, children)
                heapq.heappush(Q, (d[v], v))
                
    return d, parent, children

# --- ГЕНЕРАТОРЫ ТОПОЛОГИЙ ---
def assign_spatial_weights(G, mult_func=lambda: 1.0, min_w=0.0):
    for u, v in G.edges():
        w = math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']) * mult_func()
        G[u][v]['weight'] = max(min_w, w)

def generate_grid(n):
    side = math.ceil(math.sqrt(n))
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    if G.number_of_nodes() > n: 
        G.remove_nodes_from(list(G.nodes())[n:])
    nx.set_node_attributes(G, {node: (node%side, node//side) for node in G.nodes()}, 'pos')
    assign_spatial_weights(G, min_w=0.1)
    return G

def generate_rgg(n, radius=RGG_RADIUS):
    G = nx.convert_node_labels_to_integers(nx.random_geometric_graph(n, radius))
    assign_spatial_weights(G, lambda: random.uniform(1.0, 1.2))
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
    edges_to_add = int(G.number_of_nodes() * extra_edges_ratio)
    added = 0
    while added < edges_to_add:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            dist = math.dist(G.nodes[u]['pos'], G.nodes[v]['pos'])
            G.add_edge(u, v, weight=dist)
            added += 1
    return G

generate_ami = lambda n: generate_rgg(n, RGG_RADIUS)

# --- БАЗОВЫЕ АЛГОРИТМЫ ---
def run_dijkstra_with_ops(G, source, c):
    """Честный полный пересчет алгоритмом Дейкстры со строгим учетом операций."""
    d = {v: float('inf') for v in G.nodes()}
    parent = {v: None for v in G.nodes()}
    d[source] = 0
    Q = [(0, source)]
    
    while Q:
        dist_u, u = heapq.heappop(Q)
        c.pop += 1  # Учет извлечения (посещения узла)
        if dist_u > d[u]: continue
        
        for v in G.neighbors(u):
            c.relax += 1  # Учет релаксации ребра
            w = G[u][v]['weight']
            if d[u] + w < d[v]:
                d[v] = d[u] + w
                parent[v] = u
                heapq.heappush(Q, (d[v], v))
                
    return d, parent

def run_astar_with_ops(G, source, target, heuristic, c):
    """Эвристический алгоритм A*."""
    closed_set, g_score = set(), {n: float('inf') for n in G.nodes()}
    g_score[source] = 0
    open_set = [(heuristic(source, target, G), 0, source)]
    
    while open_set:
        f_curr, g_curr, curr = heapq.heappop(open_set)
        c.pop += 1
        if g_curr > g_score[curr]: continue
        closed_set.add(curr)
        
        if curr == target: return True
        
        for neighbor in G.neighbors(curr):
            c.relax += 1
            if neighbor in closed_set: continue
            tentative_g = g_score[curr] + G[curr][neighbor]['weight']
            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_new = tentative_g + heuristic(neighbor, target, G)
                heapq.heappush(open_set, (f_new, tentative_g, neighbor))
    raise nx.NetworkXNoPath()

# --- ИНКРЕМЕНТАЛЬНЫЙ АЛГОРИТМ ---
def process_edge_decrease(G, d, parent, children, u, v, w_new, eta, c=None):
    """Сценарий уменьшения веса ребра."""
    Q = []
    
    if d[u] + w_new < d[v] * (1 - eta):
        d[v] = d[u] + w_new
        update_parent(v, u, parent, children)
        heapq.heappush(Q, (d[v], v))
        
    if d[v] + w_new < d[u] * (1 - eta):
        d[u] = d[v] + w_new
        update_parent(u, v, parent, children)
        heapq.heappush(Q, (d[u], u))
        
    while Q:
        dist_x, x = heapq.heappop(Q)
        if c: c.pop += 1
        if dist_x > d[x]: continue
        
        for y in G.neighbors(x):
            if c: c.relax += 1
            w = G[x][y]['weight']
            if d[x] + w < d[y] * (1 - eta):
                d[y] = d[x] + w
                update_parent(y, x, parent, children)
                heapq.heappush(Q, (d[y], y))

def process_edge_increase(G, d, parent, children, u, v, eta, c=None):
    """Сценарий увеличения веса или удаления ребра."""
    roots_to_clear =[]
    if parent.get(v) == u: roots_to_clear.append(v)
    if parent.get(u) == v: roots_to_clear.append(u)
    
    if not roots_to_clear: return
    
    descendants = set()
    for root in roots_to_clear:
        stack = [root]
        while stack:
            node = stack.pop()
            if c: c.pop += 1
            descendants.add(node)
            stack.extend(children.get(node, set()))
            
    for z in descendants:
        d[z] = float('inf')
        update_parent(z, None, parent, children)
        
    boundary_candidates =[]
    for z in descendants:
        for y in G.neighbors(z):
            if c: c.relax += 1
            if y not in descendants and d[y] != float('inf'):
                w = G[y][z]['weight']
                if d[y] + w < d[z]:
                    d[z] = d[y] + w
                    update_parent(z, y, parent, children)
                    boundary_candidates.append(z)
                    
    if boundary_candidates:
        Q = [(d[z], z) for z in boundary_candidates]
        heapq.heapify(Q)
        
        while Q:
            dist_x, x = heapq.heappop(Q)
            if c: c.pop += 1
            if dist_x > d[x]: continue
            
            for y in G.neighbors(x):
                if c: c.relax += 1
                w = G[x][y]['weight']
                if d[x] + w < d[y] * (1 - eta):
                    d[y] = d[x] + w
                    update_parent(y, x, parent, children)
                    heapq.heappush(Q, (d[y], y))

# --- ОБЕРТКИ СПЕЦИФИЧЕСКИХ СЦЕНАРИЕВ ИЗМЕНЕНИЙ ---
def process_edge_removal(G, d, parent, children, u, v, eta, c=None):
    G.remove_edge(u, v)
    process_edge_increase(G, d, parent, children, u, v, eta, c)

def process_edge_addition(G, d, parent, children, u, v, w, eta, c=None):
    G.add_edge(u, v, weight=w)
    process_edge_decrease(G, d, parent, children, u, v, w, eta, c)

def process_node_failure(G, d, parent, children, node, eta, c=None):
    neighbors = list(G.neighbors(node))
    for nbr in neighbors:
        process_edge_removal(G, d, parent, children, node, nbr, eta, c)

# --- ЭКСПЕРИМЕНТЫ И СБОР МЕТРИК ---
def get_affected_metrics(G, old_p, new_p):
    A_parent = {n for n in G.nodes() if old_p.get(n) != new_p.get(n)}
    return len(A_parent)

def experiment_complexity():
    """Сравнение вычислительной сложности алгоритмов (Эксперимент 1)."""
    results = {'sizes': [], 'full':[], 'astar': [], 'inc':[]}
    generators = {'Grid': generate_grid, 'RGG': generate_rgg, 
                  'ScaleFree': generate_scale_free, 'SmallWorld': generate_small_world}
    
    for n_nodes in SIZES:
        f_ops, a_ops, i_ops = [], [],[]
        for _ in range(TRIALS):
            g_type = random.choice(list(generators.keys()))
            G = generators[g_type](n_nodes)
            if not G.edges(): continue
            
            src, target = list(G.nodes())[0], list(G.nodes())[-1]
            
            c_full = OpCounter()
            run_dijkstra_with_ops(G, src, c_full)
            f_ops.append(c_full.total_ops())
            
            c_astar = OpCounter()
            try:
                run_astar_with_ops(G, src, target, euclidean_heuristic, c_astar)
                a_ops.append(c_astar.total_ops())
            except nx.NetworkXNoPath:
                a_ops.append(c_full.total_ops())
                
            u, v = random.choice(list(G.edges()))
            old_w = G[u][v]['weight']
            new_w = old_w * random.uniform(0.5, 1.5)
            
            d_curr, parent_curr, children_curr = compute_initial_tree(G, src)
            G[u][v]['weight'] = new_w
            
            c_inc = OpCounter()
            if new_w < old_w:
                process_edge_decrease(G, d_curr, parent_curr, children_curr, u, v, new_w, 0.0, c_inc)
            else:
                process_edge_increase(G, d_curr, parent_curr, children_curr, u, v, 0.0, c_inc)
            i_ops.append(c_inc.total_ops())
                
        results['sizes'].append(n_nodes)
        results['full'].append(np.mean(f_ops))
        results['astar'].append(np.mean(a_ops))
        results['inc'].append(np.mean(i_ops))
    return results

def experiment_stability():
    """Исследование стабильности топологии и влияния порога eta (Эксперименты 2, 3, 4)."""
    results = {'eta': ETA_VALUES, 'churn': [], 'stretch':[]}
    n_nodes, n_events = 200, 30
    
    for eta in ETA_VALUES:
        ch_vals, st_vals = [],[]
        for _ in range(TRIALS):
            G = generate_grid(n_nodes)
            src = list(G.nodes())[0]
            
            d_curr, parent_curr, children_curr = compute_initial_tree(G, src)
            total_churn, total_stretch, st_count = 0, 0, 0
            
            for _ in range(n_events):
                edges = list(G.edges())
                u, v = random.choice(edges)
                old_w = G[u][v]['weight']
                new_w = old_w * random.uniform(0.85, 1.15)
                
                old_p = parent_curr.copy()
                G[u][v]['weight'] = new_w
                
                if new_w < old_w:
                    process_edge_decrease(G, d_curr, parent_curr, children_curr, u, v, new_w, eta)
                else:
                    process_edge_increase(G, d_curr, parent_curr, children_curr, u, v, eta)
                
                total_churn += get_affected_metrics(G, old_p, parent_curr)
                
                d_opt_after, _, _ = compute_initial_tree(G, src)
                
                if eta == 0.0:
                    for n in G.nodes():
                        if d_opt_after[n] != float('inf'):
                            assert math.isclose(d_curr[n], d_opt_after[n], rel_tol=1e-5)

                for node in G.nodes():
                    if node != src and d_opt_after[node] != float('inf') and d_opt_after[node] > 0:
                        stretch = max(1.0, d_curr[node] / d_opt_after[node])
                        total_stretch += stretch
                        st_count += 1
                
            ch_vals.append(total_churn / n_events)
            if st_count > 0: st_vals.append(total_stretch / st_count)
            
        results['churn'].append(np.mean(ch_vals))
        results['stretch'].append(np.mean(st_vals))
    return results

def experiment_mesh_ami():
    """Сравнение локализации обновлений на разных топологиях (Эксперимент 5)."""
    results = {'sizes': SIZES, 'mesh': [], 'ami':[]}
    for n_nodes in SIZES:
        affected = {'mesh': [], 'ami':[]}
        for topo_name, gen_func in [('mesh', generate_mesh), ('ami', generate_ami)]:
            for _ in range(TRIALS):
                G = gen_func(n_nodes)
                if not G.edges(): continue
                src = list(G.nodes())[0]
                d_curr, parent_curr, children_curr = compute_initial_tree(G, src)
                
                u, v = random.choice(list(G.edges()))
                old_w = G[u][v]['weight']
                new_w = old_w * random.uniform(0.5, 1.5)
                
                old_d = d_curr.copy()
                G[u][v]['weight'] = new_w
                if new_w < old_w:
                    process_edge_decrease(G, d_curr, parent_curr, children_curr, u, v, new_w, 0.0)
                else:
                    process_edge_increase(G, d_curr, parent_curr, children_curr, u, v, 0.0)
                
                A_dist = {n for n in G.nodes() if old_d[n] != d_curr[n]}
                affected[topo_name].append(len(A_dist))
                
        results['mesh'].append(np.mean(affected['mesh']) if affected['mesh'] else 0)
        results['ami'].append(np.mean(affected['ami']) if affected['ami'] else 0)
    return results

def experiment_stress_test():
    """Анализ границ применимости в условиях массовых отказов (Эксперимент 6)."""
    ratios = np.linspace(0.01, 0.50, 10)
    results = {'ratios': ratios * 100, 'full':[], 'inc':[]}
    n_nodes = 400
    
    for r in ratios:
        f_ops, i_ops = [],[]
        for _ in range(10):
            G = generate_grid(n_nodes)
            src = list(G.nodes())[0]
            edges = list(G.edges())
            k = int(len(edges) * r)
            if k == 0: continue
            
            removed = random.sample(edges, k)
            
            # 1. Эталонный полный пересчет (на изолированной копии)
            G_after = G.copy()
            G_after.remove_edges_from(removed)
            c_full = OpCounter()
            if G_after.number_of_edges() > 0:
                run_dijkstra_with_ops(G_after, src, c_full)
            f_ops.append(c_full.total_ops())
            
            # 2. Инкрементальный пересчет (на изолированной копии)
            G_sim = G.copy()
            d_curr, parent_curr, children_curr = compute_initial_tree(G_sim, src)
            c_inc = OpCounter()
            for u, v in removed:
                process_edge_removal(G_sim, d_curr, parent_curr, children_curr, u, v, 0.0, c_inc)
            i_ops.append(c_inc.total_ops())
            
        results['full'].append(np.mean(f_ops))
        results['inc'].append(np.mean(i_ops))
    return results

# --- ПОСТРОЕНИЕ ГРАФИКОВ ---
def plot_results(res1, res2, res3, res4):
    def save_plot(name): 
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(f'graphs/{name}.png', dpi=300); plt.close()
    
    # 1. Complexity
    plt.figure(figsize=(8, 5)); plt.title('Рис. 1. Сравнение вычислительной сложности')
    plt.plot(res1['sizes'], res1['full'], 'r-o', label='Полный пересчет (Dijkstra)')
    plt.plot(res1['sizes'], res1['astar'], 'g--s', label='A* (эвристический)')
    plt.plot(res1['sizes'], res1['inc'], 'b-^', label='Инкрементальный (Предлож.)')
    plt.xlabel('Размер сети |V|'); plt.ylabel('Вычислительные затраты (операции)')
    plt.legend(); save_plot('exp1_complex')

    # 2. Churn
    plt.figure(figsize=(8, 5)); plt.title('Рис. 2. Зависимость Route Churn от порога η')
    plt.plot(res2['eta'], res2['churn'], 'D-', color='purple')
    plt.xlabel('Порог стабилизации η'); plt.ylabel('Дрожание маршрутов (Churn)')
    save_plot('exp2_churn')
    
    # 3. Stretch
    plt.figure(figsize=(8, 5)); plt.title('Рис. 3. Зависимость качества пути от порога η')
    plt.plot(res2['eta'], res2['stretch'], '^-', color='orange')
    plt.xlabel('Порог стабилизации η'); plt.ylabel('Относительное удлинение пути (Stretch)')
    save_plot('exp2_stretch')

    # 4. Trade-off
    plt.figure(figsize=(8, 5)); plt.title('Рис. 4. Компромисс "Качество vs Стабильность"')
    sc = plt.scatter(res2['stretch'], res2['churn'], c=res2['eta'], cmap='viridis', s=100, zorder=2)
    plt.plot(res2['stretch'], res2['churn'], 'k--', alpha=0.3, zorder=1)
    plt.colorbar(sc, label='Порог η')
    plt.xlabel('Качество (Stretch, 1.0 = оптимум)'); plt.ylabel('Нестабильность (Churn)')
    save_plot('exp2_tradeoff')

    # 5. Topologies
    plt.figure(figsize=(8, 5)); plt.title('Рис. 5. Сравнение поведения на mesh и AMI топологиях')
    plt.plot(res3['sizes'], res3['mesh'], 'b-o', label='Mesh-топологии')
    plt.plot(res3['sizes'], res3['ami'], 'r-s', label='AMI-топологии')
    plt.xlabel('Размер сети |V|'); plt.ylabel('Размер области влияния |A_dist|')
    plt.legend(); save_plot('exp3_topo')

    # 6. Stress Test (С добавлением точки пересечения)
    plt.figure(figsize=(8, 5)); plt.title('Рис. 6. Границы эффективности алгоритма при массовых отказах')
    plt.plot(res4['ratios'], res4['full'], 'r-o', label='Полный пересчет (Dijkstra)')
    plt.plot(res4['ratios'], res4['inc'], 'b-s', label='Инкрементальный (Предлож.)')
    
    # Поиск точной точки пересечения линий (линейная интерполяция)
    x_vals = res4['ratios']
    y_full = res4['full']
    y_inc = res4['inc']
    intersection_x = None
    
    for i in range(len(x_vals) - 1):
        if y_inc[i] <= y_full[i] and y_inc[i+1] > y_full[i+1]:
            d0 = y_full[i] - y_inc[i]
            d1 = y_full[i+1] - y_inc[i+1]
            t = d0 / (d0 - d1)
            intersection_x = x_vals[i] + t * (x_vals[i+1] - x_vals[i])
            break
            
    if intersection_x is not None:
        plt.axvline(x=intersection_x, color='gray', linestyle='--', alpha=0.8, 
                    label=f'Граница эффективности (~{intersection_x:.1f}%)')
        
    plt.xticks(np.arange(0, max(x_vals) + 5, 5))
    plt.xlabel('Доля отказавших ребер (%)')
    plt.ylabel('Вычислительные затраты (операции)')
    plt.legend()
    save_plot('exp4_stress')

# --- ЗАПУСК ПРОГРАММЫ ---
if __name__ == "__main__":
    print("Запуск экспериментального исследования...")
    
    r1 = experiment_complexity()
    print("Эксперимент 1 (Вычислительная сложность) завершен.")
    
    r2 = experiment_stability()
    print("Эксперименты 2, 3, 4 (Стабильность, Stretch, Trade-off) завершены.")
    
    r3 = experiment_mesh_ami()
    print("Эксперимент 5 (Влияние топологии) завершен.")
    
    r4 = experiment_stress_test()
    print("Эксперимент 6 (Массовые отказы) завершен.")
    
    plot_results(r1, r2, r3, r4)
    print("\nИсследование успешно выполнено. Графики сохранены в директорию '/graphs'.")
