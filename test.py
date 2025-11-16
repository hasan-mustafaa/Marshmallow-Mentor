from edu import analyze_step
import asyncio


async def main(equation):
    response = await analyze_step(equation)
    print(response)
    return response


# Added the necessary space after \times
equation_test = "14 \\times 23 = 434 + 234"
asyncio.run(main(equation_test))
