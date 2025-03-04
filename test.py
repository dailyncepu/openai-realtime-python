import base64


def str_to_bytes(input_str):
    return input_str.encode("utf-8")


def str_to_base64(input_str):
    res = base64.b64encode(input_str.encode("utf-8")).decode("utf-8")
    return base64.b64decode(res)

def bytes_to_str(input_bytes):
    return input_bytes.decode("utf-8")

a = "高兴<##>"
b = "在这个示例中，我们首先将字符串进行 Base64 编码，然后使用"


n_a = str_to_bytes(a)
n_b = str_to_base64(b)

print(n_a,n_b)
nc = n_a + n_b

res = nc.split(b'<##>')

for e in res:
    print(bytes_to_str(e))

