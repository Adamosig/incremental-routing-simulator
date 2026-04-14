import os
import math
import random
import heapq
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# --- КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТОВ ---
random.seed(42)
np.random.seed(42)
os.makedirs('graphs', exist_ok=True)

SIZES = [100, 200, 300, 400, 500]
TRIALS = 15
ETA_VALUES = np.linspace(0, 0.3, 7)
RGG_RADIUS = 0.15
BA_M = 3
SW_K = 6
SW_P = 0.3

# --- СТРУКТУРЫ ДАННЫХ И СЧЕТЧИКИ (ИСПРАВЛЕНО: п.1 - унификация метрики) ---
class OpCounter:
    """Счётчик операций с разделением по типам."""
    def __init__(self):
        self.pq_pop = 0      # извлечения из приоритетной очереди
        self.relax = 0       # релаксации рёбер
        self.tree_scan = 0   # обход поддерева при увеличении веса
        self.boundary_scan = 0 # проверка граничных рёбер

    def total_ops(self):
        # Возвращает сумму всех операций для общего сравнения
        return self.pq_pop + self.relax + self.tree_scan + self.boundary_scan
    
    def get_metrics(self):
        # Для детального анализа
        return {
            'pq_pop': self.pq_pop,
            'relax': self.relax,
            'tree_scan': self.tree_scan,
            'boundary_scan': self.boundary_scan
        }


def euclidean_heuristic(u, v, G):
    """Евклидова эвристика для A*."""
    return math.dist(G.nodes[u]['pos'], G.nodes[v]['pos'])


def update_parent(v, new_p, parent, children):
    """Обновление родителя вершины с синхронизацией списков потомков."""
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


def compute_initial_tree(G, source, c=None):
    """Построение дерева кратчайших путей алгоритмом Дейкстры."""
    d = {v: float('inf') for v in G.nodes()}
    parent = {v: None for v in G.nodes()}
    children = {v: set() for v in G.nodes()}
    visited = {v: False for v in G.nodes()}

    d[source] = 0
    Q = [(0, source)]

    while Q:
        dist_u, u = heapq.heappop(Q)
        if visited[u] or dist_u > d[u]:
            continue

        visited[u] = True
        if c:
            c.pq_pop += 1

        for v in G.neighbors(u):
            if not visited[v]:
                if c:
                    c.relax += 1
                w = G[u][v]['weight']
                if d[u] + w < d[v]:
                    d[v] = d[u] + w
                    update_parent(v, u, parent, children)
                    heapq.heappush(Q, (d[v], v))

    return d, parent, children


# --- ГЕНЕРАТОРЫ ГРАФОВ ---
def assign_spatial_weights(G, mult_func=lambda: 1.0, min_w=0.0):
    """Присваивает веса рёбрам на основе евклидова расстояния."""
    for u, v in G.edges():
        w = math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']) * mult_func()
        G[u][v]['weight'] = max(min_w, w)


def generate_grid(n):
    """Регулярная решётка."""
    side = math.ceil(math.sqrt(n))
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    if G.number_of_nodes() > n:
        G.remove_nodes_from(list(G.nodes())[n:])
    nx.set_node_attributes(
        G,
        {node: (node % side, node // side) for node in G.nodes()},
        'pos'
    )
    assign_spatial_weights(G, min_w=0.1)
    return G


def generate_rgg(n, radius=RGG_RADIUS):
    """Случайный геометрический граф."""
    G = nx.convert_node_labels_to_integers(
        nx.random_geometric_graph(n, radius)
    )
    assign_spatial_weights(G, lambda: random.uniform(1.0, 1.2))
    return G


def generate_scale_free(n, m=BA_M):
    """Безмасштабный граф (Барабаши-Альберт)."""
    G = nx.complete_graph(n) if n <= m else nx.barabasi_albert_graph(n, m)
    nx.set_node_attributes(
        G,
        {i: (random.random(), random.random()) for i in G.nodes()},
        'pos'
    )
    assign_spatial_weights(G, lambda: random.uniform(5, 15))
    return G


def generate_small_world(n, k=SW_K, p=SW_P):
    """Граф «мир тесен» (Ваттс-Строгац)."""
    G = nx.watts_strogatz_graph(n, k, p)
    nx.set_node_attributes(
        G,
        {i: (random.random(), random.random()) for i in G.nodes()},
        'pos'
    )
    assign_spatial_weights(G, lambda: random.uniform(5, 15))
    return G


def generate_mesh(n, extra_edges_ratio=0.05):
    """Ячеистая топология на основе решётки с дополнительными локальными рёбрами."""
    G = generate_grid(n)
    nodes = list(G.nodes())
    edges_to_add = int(G.number_of_nodes() * extra_edges_ratio)
    added = 0
    radius = 2.5  # Радиус локальности для дополнительных связей
    while added < edges_to_add:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            dist = math.dist(G.nodes[u]['pos'], G.nodes[v]['pos'])
            if dist < radius * math.sqrt(2): # Ограничение по радиусу
                G.add_edge(u, v, weight=dist)
                added += 1
    return G


def generate_ami(n):
    """Топология AMI (интеллектуальные счётчики)."""
    return generate_rgg(n, RGG_RADIUS)


# --- БАЗОВЫЕ АЛГОРИТМЫ (ИСПРАВЛЕНО: п.1 - симметричный подсчет операций) ---
def run_dijkstra_with_ops(G, source, c):
    """Полный пересчёт дерева кратчайших путей."""
    d = {v: float('inf') for v in G.nodes()}
    parent = {v: None for v in G.nodes()}
    visited = {v: False for v in G.nodes()}

    d[source] = 0
    Q = [(0, source)]

    while Q:
        dist_u, u = heapq.heappop(Q)
        if visited[u] or dist_u > d[u]:
            continue

        visited[u] = True
        c.pq_pop += 1

        for v in G.neighbors(u):
            if not visited[v]:
                c.relax += 1
                w = G[u][v]['weight']
                if d[u] + w < d[v]:
                    d[v] = d[u] + w
                    parent[v] = u
                    heapq.heappush(Q, (d[v], v))

    return d, parent


def run_astar_with_ops(G, source, target, heuristic, c):
    """Поиск пути A* (используется как справочный ориентир)."""
    closed_set = set()
    g_score = {n: float('inf') for n in G.nodes()}
    g_score[source] = 0
    open_set = [(heuristic(source, target, G), 0, source)]

    while open_set:
        f_curr, g_curr, curr = heapq.heappop(open_set)
        c.pq_pop += 1
        if g_curr > g_score[curr]:
            continue
        closed_set.add(curr)

        if curr == target:
            return True

        for neighbor in G.neighbors(curr):
            c.relax += 1
            if neighbor in closed_set:
                continue
            tentative_g = g_score[curr] + G[curr][neighbor]['weight']
            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_new = tentative_g + heuristic(neighbor, target, G)
                heapq.heappush(open_set, (f_new, tentative_g, neighbor))

    raise nx.NetworkXNoPath()


# --- ИНКРЕМЕНТАЛЬНЫЙ АЛГОРИТМ (ИСПРАВЛЕНО: п.1 - симметричный подсчет) ---
def process_edge_decrease(G, d, parent, children, u, v, w_new, eta, c=None):
    """Обработка уменьшения веса ребра."""
    Q = []
    processed = set()

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
        if x in processed or dist_x > d[x]:
            continue
        processed.add(x)
        if c:
            c.pq_pop += 1

        for y in G.neighbors(x):
            if y not in processed:
                if c:
                    c.relax += 1
                w = G[x][y]['weight']
                if d[x] + w < d[y] * (1 - eta):
                    d[y] = d[x] + w
                    update_parent(y, x, parent, children)
                    heapq.heappush(Q, (d[y], y))


def process_edge_increase(G, d, parent, children, u, v, eta, c=None):
    """Обработка увеличения веса или удаления ребра."""
    roots_to_clear = []
    if parent.get(v) == u:
        roots_to_clear.append(v)
    if parent.get(u) == v:
        roots_to_clear.append(u)

    if not roots_to_clear:
        return

    descendants = set()
    for root in roots_to_clear:
        stack = [root]
        while stack:
            node = stack.pop()
            if node in descendants:
                continue
            descendants.add(node)
            if c:
                c.tree_scan += 1 # Операция обхода поддерева
            stack.extend(children.get(node, set()))

    for z in descendants:
        d[z] = float('inf')
        update_parent(z, None, parent, children)

    boundary_candidates = []
    processed_relax = set()

    for z in descendants:
        for y in G.neighbors(z):
            if y not in descendants and d[y] != float('inf'):
                edge_key = tuple(sorted([z, y]))
                if edge_key not in processed_relax:
                    if c:
                        c.boundary_scan += 1 # Операция проверки границы
                    processed_relax.add(edge_key)

                w = G[y][z]['weight']
                if d[y] + w < d[z]:
                    d[z] = d[y] + w
                    update_parent(z, y, parent, children)
                    boundary_candidates.append(z)

    if boundary_candidates:
        Q = [(d[z], z) for z in set(boundary_candidates)]
        heapq.heapify(Q)
        processed = set()

        while Q:
            dist_x, x = heapq.heappop(Q)
            if x in processed or dist_x > d[x]:
                continue
            processed.add(x)
            if c:
                c.pq_pop += 1

            for y in G.neighbors(x):
                if y not in processed:
                    edge_key = tuple(sorted([x, y]))
                    if edge_key not in processed_relax:
                        if c:
                            c.relax += 1
                        processed_relax.add(edge_key)

                    w = G[x][y]['weight']
                    if d[x] + w < d[y] * (1 - eta):
                        d[y] = d[x] + w
                        update_parent(y, x, parent, children)
                        heapq.heappush(Q, (d[y], y))


def process_edge_removal(G, d, parent, children, u, v, eta, c=None):
    """Удаление ребра."""
    G.remove_edge(u, v)
    process_edge_increase(G, d, parent, children, u, v, eta, c)


# --- СБОР МЕТРИК (ИСПРАВЛЕНО: п.8 - единый интерфейс для A_dist и A_parent) ---
def get_affected_metrics(G, old_d, new_d, old_p, new_p):
    """Возвращает размеры множеств A_dist и A_parent."""
    A_dist = {n for n in G.nodes() if not math.isclose(old_d[n], new_d[n], rel_tol=1e-7)}
    A_parent = {n for n in G.nodes() if old_p.get(n) != new_p.get(n)}
    return len(A_dist), len(A_parent)


# --- ЭКСПЕРИМЕНТЫ ---
def experiment_complexity():
    """Эксперимент 1: вычислительная сложность."""
    results = {'sizes': [], 'full': [], 'astar': [], 'inc': []}
    generators = {
        'Grid': generate_grid,
        'RGG': generate_rgg,
        'ScaleFree': generate_scale_free,
        'SmallWorld': generate_small_world
    }
    astar_pairs = 5

    for n_nodes in SIZES:
        f_ops, a_ops, i_ops = [], [], []
        for _ in range(TRIALS):
            g_type = random.choice(list(generators.keys()))
            G = generators[g_type](n_nodes)
            if not G.edges():
                continue

            src = list(G.nodes())[0]

            c_full = OpCounter()
            run_dijkstra_with_ops(G, src, c_full)
            f_ops.append(c_full.total_ops())

            a_ops_samples = []
            nodes_list = list(G.nodes())
            for _ in range(astar_pairs):
                target = random.choice(nodes_list)
                if src == target:
                    continue
                c_astar = OpCounter()
                try:
                    run_astar_with_ops(G, src, target,
                                       euclidean_heuristic, c_astar)
                    a_ops_samples.append(c_astar.total_ops())
                except nx.NetworkXNoPath:
                    pass
            a_ops.append(np.mean(a_ops_samples) if a_ops_samples
                         else c_full.total_ops())

            u, v = random.choice(list(G.edges()))
            old_w = G[u][v]['weight']
            new_w = old_w * random.uniform(0.5, 1.5)

            d_curr, parent_curr, children_curr = compute_initial_tree(G, src)
            G[u][v]['weight'] = new_w

            c_inc = OpCounter()
            if new_w < old_w:
                process_edge_decrease(G, d_curr, parent_curr, children_curr,
                                      u, v, new_w, 0.0, c_inc)
            else:
                process_edge_increase(G, d_curr, parent_curr, children_curr,
                                      u, v, 0.0, c_inc)
            i_ops.append(c_inc.total_ops())

        results['sizes'].append(n_nodes)
        results['full'].append(np.mean(f_ops))
        results['astar'].append(np.mean(a_ops))
        results['inc'].append(np.mean(i_ops))
    return results


def experiment_stability():
    """Эксперименты 2-4: влияние η на Churn и Stretch."""
    results = {
        'eta': ETA_VALUES,
        'churn_mean': [], 'churn_std': [],
        'stretch_mean': [], 'stretch_std': []
    }
    n_nodes, n_events = 200, 30

    for eta in ETA_VALUES:
        ch_all, st_all = [], []

        for _ in range(TRIALS):
            G = generate_grid(n_nodes)
            src = list(G.nodes())[0]

            d_curr, parent_curr, children_curr = compute_initial_tree(G, src)
            total_churn, total_stretch, st_count = 0, 0, 0

            for _ in range(n_events):
                u, v = random.choice(list(G.edges()))
                old_w = G[u][v]['weight']
                new_w = old_w * random.uniform(0.85, 1.15)

                old_p = parent_curr.copy()
                old_d = d_curr.copy()
                G[u][v]['weight'] = new_w

                if new_w < old_w:
                    process_edge_decrease(G, d_curr, parent_curr, children_curr,
                                          u, v, new_w, eta)
                else:
                    process_edge_increase(G, d_curr, parent_curr, children_curr,
                                          u, v, eta)

                _, A_parent_len = get_affected_metrics(G, old_d, d_curr, old_p, parent_curr)
                total_churn += A_parent_len
                
                d_opt_after, _, _ = compute_initial_tree(G, src)

                if eta == 0.0:
                    for n in G.nodes():
                        if d_opt_after[n] != float('inf'):
                            assert math.isclose(d_curr[n], d_opt_after[n],
                                                rel_tol=1e-5)

                for node in G.nodes():
                    if (node != src and d_opt_after[node] != float('inf')
                            and d_opt_after[node] > 0):
                        stretch = max(1.0, d_curr[node] / d_opt_after[node])
                        total_stretch += stretch
                        st_count += 1

            ch_all.append(total_churn / n_events)
            if st_count > 0:
                st_all.append(total_stretch / st_count)

        results['churn_mean'].append(np.mean(ch_all))
        results['churn_std'].append(np.std(ch_all))
        results['stretch_mean'].append(np.mean(st_all))
        results['stretch_std'].append(np.std(st_all))

    return results


def experiment_mesh_ami():
    """Эксперимент 5: размер области влияния для Mesh и AMI."""
    results = {'sizes': SIZES, 'mesh_A_dist': [], 'ami_A_dist': []}
    for n_nodes in SIZES:
        affected_mesh, affected_ami = [], []
        for topo_name, gen_func in [('mesh', generate_mesh),
                                    ('ami', generate_ami)]:
            for _ in range(TRIALS):
                G = gen_func(n_nodes)
                if not G.edges():
                    continue
                src = list(G.nodes())[0]
                d_curr, parent_curr, children_curr = compute_initial_tree(
                    G, src
                )

                u, v = random.choice(list(G.edges()))
                old_w = G[u][v]['weight']
                new_w = old_w * random.uniform(0.5, 1.5)

                old_d = d_curr.copy()
                old_p = parent_curr.copy()
                G[u][v]['weight'] = new_w
                if new_w < old_w:
                    process_edge_decrease(G, d_curr, parent_curr, children_curr,
                                          u, v, new_w, 0.0)
                else:
                    process_edge_increase(G, d_curr, parent_curr, children_curr,
                                          u, v, 0.0)

                A_dist_len, _ = get_affected_metrics(G, old_d, d_curr, old_p, parent_curr)
                
                if topo_name == 'mesh':
                    affected_mesh.append(A_dist_len)
                else:
                    affected_ami.append(A_dist_len)

        results['mesh_A_dist'].append(np.mean(affected_mesh) if affected_mesh else 0)
        results['ami_A_dist'].append(np.mean(affected_ami) if affected_ami else 0)
    return results


def experiment_stress_test():
    """Эксперимент 6: границы применимости при массовых отказах."""
    ratios = np.linspace(0.005, 0.50, 20)
    n_nodes = 400
    n_outer = 5
    n_inner = 10

    all_intersections = []
    avg_full = np.zeros(len(ratios))
    avg_inc = np.zeros(len(ratios))
    all_full_curves = []
    all_inc_curves = []

    print("  Запуск стресс-теста...")

    for _ in range(n_outer):
        f_means, i_means = [], []

        for r in ratios:
            f_ops, i_ops = [], []

            for _ in range(n_inner):
                G = generate_grid(n_nodes)
                while not nx.is_connected(G):
                    G = generate_grid(n_nodes)

                src = list(G.nodes())[0]
                edges = list(G.edges())
                k = int(len(edges) * r)
                if k == 0:
                    continue

                removed = random.sample(edges, k)

                G_after = G.copy()
                G_after.remove_edges_from(removed)
                c_full = OpCounter()
                if G_after.number_of_edges() > 0:
                    run_dijkstra_with_ops(G_after, src, c_full)
                f_ops.append(c_full.total_ops())

                G_sim = G.copy()
                d_curr, parent_curr, children_curr = compute_initial_tree(
                    G_sim, src
                )
                c_inc = OpCounter()
                for u, v in removed:
                    if G_sim.has_edge(u, v):
                        process_edge_removal(G_sim, d_curr, parent_curr,
                                             children_curr, u, v, 0.0, c_inc)
                i_ops.append(c_inc.total_ops())

            f_means.append(np.mean(f_ops) if f_ops else 0)
            i_means.append(np.mean(i_ops) if i_ops else 0)

        all_full_curves.append(f_means)
        all_inc_curves.append(i_means)

        avg_full += np.array(f_means)
        avg_inc += np.array(i_means)

        x_vals = ratios * 100
        y_full = np.array(f_means)
        y_inc = np.array(i_means)

        for i in range(len(x_vals) - 1):
            if y_inc[i] <= y_full[i] and y_inc[i + 1] > y_full[i + 1]:
                d0 = y_full[i] - y_inc[i]
                d1 = y_full[i + 1] - y_inc[i + 1]
                if abs(d0 - d1) > 1e-10:
                    t = d0 / (d0 - d1)
                    intersection = x_vals[i] + t * (x_vals[i + 1] - x_vals[i])
                    all_intersections.append(intersection)
                    break

    avg_full /= n_outer
    avg_inc /= n_outer

    full_std = np.std(all_full_curves, axis=0)
    inc_std = np.std(all_inc_curves, axis=0)

    if all_intersections:
        mean_intersect = np.mean(all_intersections)
        std_intersect = np.std(all_intersections)
        # ИСПРАВЛЕНО: п.4 - Расчет 95% доверительного интервала вместо просто std
        ci_intersect = stats.t.interval(0.95, len(all_intersections)-1, loc=mean_intersect, scale=stats.sem(all_intersections))
        print(f"  Граница эффективности: {mean_intersect:.2f}% (95% CI: [{ci_intersect[0]:.2f}%, {ci_intersect[1]:.2f}%])")
    else:
        mean_intersect = None
        ci_intersect = (None, None)
        print("  Внимание: точка пересечения не найдена")

    results = {
        'ratios': ratios * 100,
        'full_mean': avg_full.tolist(),
        'full_std': full_std.tolist(),
        'inc_mean': avg_inc.tolist(),
        'inc_std': inc_std.tolist()
    }
    intersection_stats = {
        'mean': mean_intersect,
        'std': std_intersect if all_intersections else None,
        'ci': ci_intersect,
        'all_values': all_intersections
    }
    return results, intersection_stats


# --- ПОСТРОЕНИЕ ГРАФИКОВ (ИСПРАВЛЕНО: п.6, п.9) ---
def plot_results(res1, res2, res3, res4, intersect_stats):
    """Формирование и сохранение всех графиков."""
    def save_plot(name):
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'graphs/{name}.png', dpi=300)
        plt.close()

    # Рис. 1: Вычислительная сложность (A* как справочный ориентир)
    plt.figure(figsize=(8, 5))
    plt.title('Рис. 1. Сравнение вычислительной сложности')
    plt.plot(res1['sizes'], res1['full'], 'r-o', label='Полный пересчет')
    plt.plot(res1['sizes'], res1['astar'], 'g--s', label='A* (справочно)')
    plt.plot(res1['sizes'], res1['inc'], 'b-^', label='Инкрементальный')
    plt.xlabel('Размер сети |V|')
    plt.ylabel('Вычислительные затраты (операции)')
    plt.legend()
    save_plot('exp1_complex')

    # Рис. 2: Churn
    plt.figure(figsize=(8, 5))
    plt.title('Рис. 2. Зависимость дрожания маршрутов от порога η')
    x = res2['eta']
    y = res2['churn_mean']
    std = res2['churn_std']
    plt.plot(x, y, 'D-', color='purple', linewidth=2, markersize=8)
    plt.fill_between(x, np.array(y) - np.array(std),
                     np.array(y) + np.array(std), alpha=0.25, color='purple')
    plt.xlabel('Порог стабилизации η')
    plt.ylabel('Дрожание маршрутов (Churn)')
    save_plot('exp2_churn')

    # Рис. 3: Stretch
    plt.figure(figsize=(8, 5))
    plt.title('Рис. 3. Зависимость качества пути от порога η')
    x = res2['eta']
    y = res2['stretch_mean']
    std = res2['stretch_std']
    plt.plot(x, y, '^-', color='orange', linewidth=2, markersize=8)
    plt.fill_between(x, np.array(y) - np.array(std),
                     np.array(y) + np.array(std), alpha=0.25, color='orange')
    plt.xlabel('Порог стабилизации η')
    plt.ylabel('Относительное удлинение пути (Stretch)')
    save_plot('exp2_stretch')

    # Рис. 4: Trade-off
    plt.figure(figsize=(10, 7))
    plt.title('Рис. 4. Компромисс "Качество vs Стабильность"')
    x_vals = res2['stretch_mean']
    y_vals = res2['churn_mean']
    eta_vals = res2['eta']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    for i, eta in enumerate(eta_vals):
        plt.scatter(x_vals[i], y_vals[i], marker=markers[i], s=50,
                    facecolors='white', edgecolors='black', linewidth=1.5,
                    zorder=3, label=f'η = {eta:.2f}')

    plt.plot(x_vals, y_vals, 'k--', alpha=0.2, linewidth=0.5, zorder=1)
    plt.xlabel('Качество (Stretch, 1.0 = оптимум)')
    plt.ylabel('Нестабильность (Churn)')
    plt.xticks(np.arange(1.000, 1.010, 0.001))
    plt.legend(loc='upper right', title='Значения η', framealpha=0.9)

    x_margin = (max(x_vals) - min(x_vals)) * 0.18
    y_margin = (max(y_vals) - min(y_vals)) * 0.18
    plt.xlim(min(x_vals) - x_margin, max(x_vals) + x_margin * 1.5)
    plt.ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)
    save_plot('exp2_tradeoff')

    # Рис. 5: Mesh vs AMI (ИСПРАВЛЕНО: п.3 - явное указание на размер области влияния)
    plt.figure(figsize=(8, 5))
    plt.title('Рис. 5. Размер области влияния на mesh и AMI топологиях')
    plt.plot(res3['sizes'], res3['mesh_A_dist'], 'b-o', label='Mesh-топологии')
    plt.plot(res3['sizes'], res3['ami_A_dist'], 'r-s', label='AMI-топологии')
    plt.xlabel('Размер сети |V|')
    plt.ylabel('Размер области влияния |A_dist|')
    plt.legend()
    save_plot('exp3_topo')

    # Рис. 6: Stress Test (ИСПРАВЛЕНО: п.4, п.5, п.9 - CI вместо std, ограничение Grid)
    plt.figure(figsize=(9, 6))
    plt.title('Рис. 6. Границы эффективности при серии отказов (регулярная решётка)')
    x = res4['ratios']

    plt.plot(x, res4['full_mean'], 'r-o', label='Полный пересчет', linewidth=2)
    plt.fill_between(x,
                     np.array(res4['full_mean']) - np.array(res4['full_std']),
                     np.array(res4['full_mean']) + np.array(res4['full_std']),
                     alpha=0.2, color='red')

    plt.plot(x, res4['inc_mean'], 'b-s', label='Инкрементальный', linewidth=2)
    plt.fill_between(x,
                     np.array(res4['inc_mean']) - np.array(res4['inc_std']),
                     np.array(res4['inc_mean']) + np.array(res4['inc_std']),
                     alpha=0.2, color='blue')

    if intersect_stats['mean'] is not None:
        mean_val = intersect_stats['mean']
        ci = intersect_stats['ci']
        plt.axvline(x=mean_val, color='gray', linestyle='--', linewidth=1.5,
                    label=f'Граница эффективности ({mean_val:.2f}%, 95% CI: [{ci[0]:.2f}%, {ci[1]:.2f}%])')
        plt.axvspan(ci[0], ci[1], alpha=0.12, color='gray')

        y_max = plt.ylim()[1]
        plt.annotate(f'{mean_val:.2f}%', xy=(mean_val, y_max * 0.5),
                     xytext=(mean_val + 2, y_max * 0.6),
                     arrowprops=dict(arrowstyle='->', color='gray'),
                     fontsize=10, color='gray')

    plt.xticks(np.arange(0, max(x) + 5, 5))
    plt.xlabel('Доля отказавших ребер (%)')
    plt.ylabel('Вычислительные затраты (операции)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    save_plot('exp4_stress')


# --- ТОЧКА ВХОДА ---
if __name__ == "__main__":
    print("=== ЗАПУСК ЭКСПЕРИМЕНТОВ ===")

    print("\n[1/4] Вычислительная сложность...")
    r1 = experiment_complexity()

    print("\n[2/4] Стабильность (Churn, Stretch)...")
    r2 = experiment_stability()

    print("\n[3/4] Топологии Mesh и AMI (размер области влияния)...")
    r3 = experiment_mesh_ami()

    print("\n[4/4] Стресс-тест (границы применимости на решётке)...")
    r4, intersect_stats = experiment_stress_test()

    print("\n=== ПОСТРОЕНИЕ ГРАФИКОВ ===")
    plot_results(r1, r2, r3, r4, intersect_stats)

    print("\n=== ГОТОВО ===")
    print("Графики сохранены в './graphs/'")

    if intersect_stats['mean'] is not None:
        idx = 3  # η = 0.15
        churn_reduction = ((r2['churn_mean'][0] - r2['churn_mean'][idx])
                           / r2['churn_mean'][0] * 100)
        stretch_value = (r2['stretch_mean'][idx] - 1.0) * 100
        print(f"\nГраница эффективности: {intersect_stats['mean']:.2f}% "
              f"(95% CI: [{intersect_stats['ci'][0]:.2f}%, {intersect_stats['ci'][1]:.2f}%])")
        print(f"Снижение Churn при η=0.15: ~{churn_reduction:.1f}%")
        print(f"Удлинение пути при η=0.15: ~{stretch_value:.2f}%")

    input("\nНажмите Enter для завершения...")
