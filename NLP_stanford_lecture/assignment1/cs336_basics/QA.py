# print(chr(2))
# chr(1)
# chr(29275)
# print(chr(29277))

# # question 1, chr(0) return nothing it is the null character
# print("this is a test" +chr(0) +"string")


# test_string = "hello! こんにちは!"
# utf8_encoded = test_string.encode("utf-8")
# print(utf8_encoded)
# #b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
# print(type(utf8_encoded))
# #<class 'bytes'>
# print(list(utf8_encoded))
# #[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
# print(len(list(utf8_encoded)))
# #23
# print(len(test_string))
# #13
# print(len(utf8_encoded))
# #23
# print(utf8_encoded.decode('utf-8'))
# #hello! こんにちは!

# """
# Using UTF-8 encoded bytes use less memory and it is then most cost efficient than UTF-16 and UTF 32
# as we can see below on the same example. hence, best for memory usage + supported by every modern systems
# """
# utf16_encoded = test_string.encode("utf16")
# utf32_encoded = test_string.encode("utf32")

# print("result in different unicode standard")
# print(utf8_encoded)

# # b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
# print(utf16_encoded)
# # b'\xff\xfeh\x00e\x00l\x00l\x00o\x00!\x00 \x00S0\x930k0a0o0!\x00'
# print(utf32_encoded)
# #b'\xff\xfe\x00\x00h\x00\x00\x00e\x00\x00\x00l\x00\x00\x00l\x00\x00\x00o\x00\x00\x00!\x00\x00\x00 \x00\x00\x00S0\x00\x00\x930\x00\x00k0\x00\x00a0\x00\x00o0\x00\x00!\x00\x00\x00'


# def decode_utf8_bytes_to_str_wrong(bytestring : bytes):
#     return bytestring.decode('utf-8')

# print(decode_utf8_bytes_to_str_wrong('hello'.encode("utf-8")))


# import regex as re
# # test of regex
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# print(re.findall(PAT, "some text that i'll pre-tokenize"))
# print(re.finditer(PAT, "some text that i'll pre-tokenize"))

# for match in re.finditer(PAT, "some text that i'll pre-tokenize"):
#     print(f"'{match.group()}' at position {match.start()}-{match.end()}")
