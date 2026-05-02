import math
import random
from dataclasses import dataclass


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


def egcd(a, b):
    if a == 0:
        return b, 0, 1
    g, y, x = egcd(b % a, a)
    return g, x - (b // a) * y, y


def modinv(a, m):
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % m


def is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    limit = int(math.sqrt(n)) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True


def random_prime(low=1000, high=5000):
    while True:
        candidate = random.randint(low, high)
        if candidate % 2 == 0:
            candidate += 1
        if is_prime(candidate):
            return candidate


@dataclass
class PublicKey:
    n: int
    g: int


@dataclass
class PrivateKey:
    lam: int
    mu: int


def generate_paillier_keypair():
    p = random_prime()
    q = random_prime()
    while q == p:
        q = random_prime()

    n = p * q
    lam = lcm(p - 1, q - 1)
    g = n + 1
    n_sq = n * n
    x = pow(g, lam, n_sq)
    l_value = (x - 1) // n
    mu = modinv(l_value, n)
    return PublicKey(n=n, g=g), PrivateKey(lam=lam, mu=mu)


def encrypt(public_key, plaintext):
    n = public_key.n
    n_sq = n * n
    if not 0 <= plaintext < n:
        raise ValueError("plaintext must be in [0, n)")

    r = random.randint(1, n - 1)
    while math.gcd(r, n) != 1:
        r = random.randint(1, n - 1)

    c1 = pow(public_key.g, plaintext, n_sq)
    c2 = pow(r, n, n_sq)
    return (c1 * c2) % n_sq


def decrypt(public_key, private_key, ciphertext):
    n = public_key.n
    n_sq = n * n
    x = pow(ciphertext, private_key.lam, n_sq)
    l_value = (x - 1) // n
    return (l_value * private_key.mu) % n


def add_encrypted(public_key, ciphertexts):
    n_sq = public_key.n * public_key.n
    result = 1
    for ciphertext in ciphertexts:
        result = (result * ciphertext) % n_sq
    return result


def scale_for_demo(value, scale=1000):
    return int(round(value * scale))


def descale_from_demo(value, scale=1000):
    return value / scale


def run_demo():
    random.seed(42)
    public_key, private_key = generate_paillier_keypair()

    client_updates = {
        "client_1": [0.125, -0.330, 0.842],
        "client_2": [0.210, 0.155, -0.120],
        "client_3": [-0.085, 0.410, 0.278],
    }

    scale = 1000
    offset = 5000

    print("=== Paillier Additive Homomorphic Prototype ===")
    print("This is an educational prototype for thesis demonstration only.")
    print(f"Public modulus n: {public_key.n}")
    print()

    encrypted_vectors = {}
    for client_id, vector in client_updates.items():
        encoded = [scale_for_demo(v, scale) + offset for v in vector]
        encrypted_vectors[client_id] = [encrypt(public_key, x) for x in encoded]
        print(f"{client_id} plaintext update: {vector}")
        print(f"{client_id} encoded update:   {encoded}")
        print()

    aggregated_cipher = []
    num_dims = len(next(iter(encrypted_vectors.values())))
    for dim in range(num_dims):
        dim_ciphertexts = [encrypted_vectors[client_id][dim] for client_id in encrypted_vectors]
        aggregated_cipher.append(add_encrypted(public_key, dim_ciphertexts))

    decrypted_sum = [decrypt(public_key, private_key, c) for c in aggregated_cipher]
    decoded_sum = [descale_from_demo(v - offset * len(client_updates), scale) for v in decrypted_sum]

    plaintext_sum = []
    for dim in range(num_dims):
        plaintext_sum.append(round(sum(client_updates[client_id][dim] for client_id in client_updates), 3))

    print("Server performs encrypted aggregation without seeing plaintext updates.")
    print(f"Decrypted aggregated sum: {decoded_sum}")
    print(f"Plaintext aggregated sum: {plaintext_sum}")
    print()

    matches = all(abs(a - b) < 1e-6 for a, b in zip(decoded_sum, plaintext_sum))
    print(f"Aggregation correctness check: {matches}")
    if not matches:
        raise RuntimeError("Homomorphic aggregation result mismatch")


if __name__ == "__main__":
    run_demo()
